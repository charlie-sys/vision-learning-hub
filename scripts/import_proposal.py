# -*- coding: utf-8 -*-
"""Import the proposal workbook export as a cross-cutting, deduplicated collection."""
from pathlib import Path
from urllib.parse import urlsplit
import json
import re
import unicodedata

ROOT=Path(__file__).resolve().parents[1]
GROUP='07_申博研究计划'


def normalized_title(title):
    title=re.sub(r'（[^）]*）\s*$','',title)
    return re.sub(r'[^a-z0-9]','',unicodedata.normalize('NFKC',title).lower())


def arxiv_id(url):
    parts=urlsplit(url)
    if parts.hostname not in ('arxiv.org','www.arxiv.org'): return ''
    match=re.fullmatch(r'/(?:abs|pdf)/(\d{4}\.\d{4,5})(?:v\d+)?(?:\.pdf)?',parts.path)
    return match.group(1) if match else ''


def main():
    source=json.loads((ROOT/'data/proposal_source.json').read_text(encoding='utf-8'))
    metadata=json.loads((ROOT/'data/proposal_source_metadata.json').read_text(encoding='utf-8'))
    rows=list(source['sheets'].values())[0][1:]
    assert len(rows)==25
    papers=json.loads((ROOT/'data/papers.json').read_text(encoding='utf-8'))
    for p in papers: p.pop('proposal_entry',None)
    by_arxiv={p['arxiv']:p for p in papers if p.get('arxiv')}
    by_url={p['source'].rstrip('/'):p for p in papers}
    by_title={normalized_title(p['title']):p for p in papers}
    mapping=[]
    for row in rows:
        number,title,url,year_source,category,relevance=row
        aid=arxiv_id(url)
        match=by_arxiv.get(aid) if aid else None
        match=match or by_url.get(url.rstrip('/')) or by_title.get(normalized_title(title))
        reused=bool(match and match.get('collection')!='proposal')
        if not match:
            meta=metadata.get(url,{})
            canonical=meta.get('title') or re.sub(r'（[^）]*）\s*$','',title)
            resource=urlsplit(url).hostname=='github.com'
            date=meta.get('published','')
            y=re.search(r'(?:19|20)\d{2}',date or year_source)
            prefix=canonical.split(':',1)[0]
            alias=re.search(r'（([^）]+)）\s*$',title)
            short=alias.group(1).replace('提出 ','').replace('提出','') if alias and not resource else prefix
            if len(short)>45: short=canonical
            pdfs=[meta['pdf_url']] if meta.get('pdf_url') else ([f'https://arxiv.org/pdf/{aid}'] if aid else [])
            if resource: pdfs=[]
            match={'id':max(p['id'] for p in papers)+1,'collection':'proposal','group':GROUP,
                'record_type':'resource' if resource else 'paper','short':short,'title':canonical,'title_zh':'',
                'authors':meta.get('authors',[]),'year':int(y.group(0)) if y else None,
                'venue':year_source,'level':'未分级','note':'原表研究关联：'+relevance,
                'arxiv':aid,'doi':meta.get('doi',''),'source':url,'pdf_urls':pdfs,
                'filename':f'Proposal_{aid or str(number)}.pdf' if pdfs else '',
                'metadata_provenance':meta.get('verification','用户提供的申博检索清单；未补写作者及PDF地址')}
            papers.append(match)
            if aid: by_arxiv[aid]=match
            by_url[url.rstrip('/')]=match
            by_title[normalized_title(canonical)]=match
        match['proposal_entry']={'row_id':number,'title':title,'link':url,'year_source':year_source,
                                 'category':category,'relevance':relevance}
        mapping.append({'row_id':number,'paper_id':match['id'],'title':title,'link':url,
                        'category':category,'reused':reused,'record_type':match.get('record_type','paper')})
    assert len(mapping)==len({r['paper_id'] for r in mapping})==25
    assert sum(r['reused'] for r in mapping)==13
    assert sum(r['record_type']=='resource' for r in mapping)==1
    (ROOT/'data/papers.json').write_text(json.dumps(papers,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    report={k:source[k] for k in ('imported_on','workbook_filename','workbook_sha256')}
    report.update(entry_count=25,paper_count=24,resource_count=1,reused_count=13,new_papers=11,
                  writing_guide_count=6,framework_row_count=10,records=mapping)
    (ROOT/'data/proposal_import_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    bibfile=ROOT/'data/references.bib'
    bib=bibfile.read_text(encoding='utf-8').split('% BEGIN PROPOSAL CATALOG')[0].rstrip()+'\n\n% BEGIN PROPOSAL CATALOG\n'
    escape=lambda value:str(value).replace('&',r'\&').replace('%',r'\%').replace('_',r'\_')
    for p in papers:
        if p.get('collection')!='proposal': continue
        fields={'title':p['title'],'url':p['source']}
        if p['authors']: fields['author']=' and '.join(p['authors'])
        if p['year']: fields['year']=p['year']
        if p.get('arxiv'): fields.update(eprint=p['arxiv'],archivePrefix='arXiv')
        if p.get('doi'): fields['doi']=p['doi']
        if p.get('record_type')=='resource': fields['note']='Literature and code resource repository'
        bib+='@misc{Proposal'+str(p['id'])+',\n'+''.join(f'  {k} = {{{escape(v)}}},\n' for k,v in fields.items())+'}\n\n'
    bibfile.write_text(bib.rstrip()+'\n',encoding='utf-8')
    print(f'Proposal collection: 24 papers + 1 resource; 13 reused, 11 new papers. Catalog: {len(papers)} entries.')


if __name__=='__main__': main()
