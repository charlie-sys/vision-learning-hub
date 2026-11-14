# -*- coding: utf-8 -*-
"""Merge the supplied spreadsheet export into the catalog without renumbering the core papers."""
from pathlib import Path
import csv
import json
import re

ROOT = Path(__file__).resolve().parents[1]
GROUP = "06_视觉幻觉"


def main():
    source = ROOT / "data/hallucination_source.csv"
    with source.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    supplement = json.loads((ROOT / "data/hallucination_source_metadata.json").read_text(encoding="utf-8"))
    assert len(rows) == 79 and [int(r['序号']) for r in rows] == list(range(1,80))
    catalog = json.loads((ROOT / "data/papers.json").read_text(encoding="utf-8"))
    foundation = [p for p in catalog if p.get('collection') != 'hallucination' and p['group'] != GROUP]
    assert len(foundation) == 30
    for p in foundation:
        p['collection'] = 'foundation'
    added = []
    for row in rows:
        arxiv = re.fullmatch(r'https://arxiv\.org/abs/(\d{4}\.\d{4,5})',row['论文地址（arXiv）']).group(1)
        extra = supplement[arxiv]
        code_urls = re.findall(r'https?://[^\s<>"\]};,)]+', row['代码/项目链接'])
        added.append({
            'id': 30 + int(row['序号']), 'collection':'hallucination', 'group': GROUP,
            'subcategory':row['研究主题分类'], 'spreadsheet_id':int(row['序号']),
            'short': row['中文译名'], 'title':row['论文题目（英文）'], 'title_zh':row['中文译名'],
            'authors':[a.strip() for a in row['作者'].split(',') if a.strip()],
            'year': int(row['发表时间'][:4]), 'published_month':row['发表时间'],
            'venue':row['正式发表载体'], 'level':'未分级', 'note':row['中文核心要点'],
            'keywords':row['关键词'], 'abstract_en':row['摘要（arXiv 原文）'], 'abstract_zh':row['摘要（中文译文）'],
            'arxiv':arxiv, 'arxiv_versioned':extra['arxiv_versioned'], 'doi':extra.get('doi',''),
            'first_submitted':extra['first_submitted'], 'source':row['论文地址（arXiv）'],
            'pdf_urls':[f'https://arxiv.org/pdf/{extra["arxiv_versioned"]}',f'https://arxiv.org/pdf/{arxiv}'],
            'filename': f'{30+int(row["序号"]):03d}_Hallucination_{arxiv}.pdf',
            'code_urls':list(dict.fromkeys(code_urls)), 'publication_type':row['发表载体类型'],
            'ccf':row['CCF 推荐等级'], 'cas_quartile':row['中科院分区'], 'jcr_quartile':row['JCR 分区'],
            'impact_factor':row['影响因子'], 'publication_notes':row['分区与收录说明'],
            'arxiv_acceptance_note':row['arXiv 录用备注（原始）'], 'arxiv_category':row['arXiv 主分类'],
            'metadata_provenance':'用户提供的视觉幻觉论文清单（Excel）；导入日期2026-11-14',
        })
    papers = foundation + added
    assert len({p['id'] for p in papers}) == len(papers)
    assert len({p['arxiv'] for p in papers if p.get('arxiv')}) == sum(bool(p.get('arxiv')) for p in papers)
    (ROOT / 'data/papers.json').write_text(json.dumps(papers,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    bibfile = ROOT / 'data/references.bib'
    bib = bibfile.read_text(encoding='utf-8').split('% BEGIN HALLUCINATION CATALOG')[0].rstrip()+'\n\n% BEGIN HALLUCINATION CATALOG\n'
    escape = lambda s: str(s).replace('\\',r'\textbackslash{}').replace('&',r'\&').replace('%',r'\%').replace('_',r'\_')
    for p in added:
        fields = {'title':p['title'], 'author':' and '.join(p['authors']), 'year':p['year'],
                  'eprint':p['arxiv'], 'archivePrefix':'arXiv', 'primaryClass':p['arxiv_category'], 'url':p['source']}
        if p['doi']: fields['doi'] = p['doi']
        bib += '@misc{Hallucination'+p['arxiv'].replace('.','_')+',\n'+''.join(f'  {k} = {{{escape(v)}}},\n' for k,v in fields.items())+'}\n\n'
    bibfile.write_text(bib.rstrip()+'\n',encoding='utf-8')
    print(f'Imported {len(added)} spreadsheet papers; retained {len(foundation)} foundation papers; total {len(papers)}')


if __name__ == '__main__':
    main()
