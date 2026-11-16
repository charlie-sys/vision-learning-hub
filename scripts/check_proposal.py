# -*- coding: utf-8 -*-
"""Verify all workbook rows, membership reuse, resource semantics and public exports."""
from pathlib import Path
from collections import Counter
import csv
import json
from bs4 import BeautifulSoup

ROOT=Path(__file__).resolve().parents[1]
source=json.loads((ROOT/'data/proposal_source.json').read_text(encoding='utf-8'))
sheets=list(source['sheets'].values())
rows=sheets[0][1:]
papers=json.loads((ROOT/'data/papers.json').read_text(encoding='utf-8'))
members=sorted((p for p in papers if p.get('proposal_entry')),key=lambda p:p['proposal_entry']['row_id'])
assert len(rows)==len(members)==25
assert len(sheets[1][1:])==6 and len(sheets[2][1:])==10
assert len(papers)==121 and sum(p.get('record_type')=='resource' for p in papers)==1
assert sum(p.get('collection')=='proposal' for p in papers)==12
assert sum(p.get('collection')!='proposal' for p in members)==13
assert len({p['source'].rstrip('/') for p in papers})==len(papers)
assert len({p['arxiv'] for p in papers if p.get('arxiv')})==sum(bool(p.get('arxiv')) for p in papers)
soup=BeautifulSoup((ROOT/'docs/index.html').read_text(encoding='utf-8'),'html.parser')
assert soup.select_one('#g option[value="07_申博研究计划"]')
assert len(soup.select('article[data-proposal-topic]:not([data-proposal-topic=""])'))==25
assert {o['value'] for o in soup.select('#p option') if o['value']}=={r[4] for r in rows}
for row,p in zip(rows,members):
    entry=p['proposal_entry']
    assert [entry[k] for k in ('row_id','title','link','year_source','category','relevance')]==row
    card=soup.select_one(f'#paper-{p["id"]}')
    assert card['data-proposal-topic']==row[4]
    assert row[5] in card.select_one('.proposal-details').get_text()
    assert row[3] in card.select_one('.proposal-details').get_text()
    assert card.select_one('.proposal-details a')['href']==row[2]
    if not p.get('pdf_urls'):
        assert not card.select_one('.translation-trigger') and not card.select_one('.download')
        assert card.select_one('footer a')['href']==p['source']
    if p.get('record_type')=='resource':
        assert '打开资源仓库' in card.get_text() and not p['pdf_urls']
guide=BeautifulSoup((ROOT/'docs/proposal.html').read_text(encoding='utf-8'),'html.parser')
assert len(guide.select('table'))==3
assert [len(t.select('tbody tr')) for t in guide.select('table')]==[25,6,10]
for row in sheets[1][1:]: assert guide.find('a',href=row[2])
export=list(csv.reader((ROOT/'docs/proposal_source.csv').open(encoding='utf-8-sig',newline='')))
assert [[str(v) if v is not None else '' for v in r] for r in sheets[0]]==export
assert json.loads((ROOT/'docs/proposal_source.json').read_text(encoding='utf-8'))==source
report=json.loads((ROOT/'data/proposal_import_report.json').read_text(encoding='utf-8'))
assert report['reused_count']==13 and report['new_papers']==11
assert [r['paper_id'] for r in report['records']]==[p['id'] for p in members]
print('Proposal: 24 papers + 1 resource, 13 reused, 11 new papers; 6 guides, 10 framework rows and exact workbook fields: OK')
