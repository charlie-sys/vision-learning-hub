# -*- coding: utf-8 -*-
"""Check every supplied spreadsheet row against catalog, cards, filters and downloads."""
from pathlib import Path
from collections import Counter
import csv
import json
from urllib.parse import urlsplit
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
rows = list(csv.DictReader((ROOT/'data/hallucination_source.csv').open(encoding='utf-8-sig',newline='')))
papers = json.loads((ROOT/'data/papers.json').read_text(encoding='utf-8'))
hallucination = [p for p in papers if p.get('collection')=='hallucination']
assert len(rows)==len(hallucination)==79
assert len({p['arxiv'] for p in papers if p.get('arxiv')})==sum(bool(p.get('arxiv')) for p in papers)
soup = BeautifulSoup((ROOT/'docs/index.html').read_text(encoding='utf-8'),'html.parser')
assert len(soup.select('article[data-group="06_视觉幻觉"]'))==79
assert len(soup.select('article'))==len(papers)
assert soup.select_one('#g option[value="06_视觉幻觉"]')
topics = {p['subcategory'] for p in hallucination}
assert len(topics)==9 and {o['value'] for o in soup.select('#s option') if o['value']}==topics
for row,paper in zip(rows,hallucination):
    assert paper['spreadsheet_id']==int(row['序号']) and paper['id']==30+int(row['序号'])
    for field,column in [('title','论文题目（英文）'),('title_zh','中文译名'),('keywords','关键词'),
                         ('abstract_zh','摘要（中文译文）'),('abstract_en','摘要（arXiv 原文）'),
                         ('note','中文核心要点'),('subcategory','研究主题分类'),('source','论文地址（arXiv）'),
                         ('venue','正式发表载体'),('ccf','CCF 推荐等级'),('cas_quartile','中科院分区'),
                         ('jcr_quartile','JCR 分区'),('impact_factor','影响因子')]:
        assert str(paper[field])==row[column], (paper['id'],field)
    card = soup.select_one(f'article#paper-{paper["id"]}')
    assert card.select_one('.paper-details') and paper['abstract_zh'] in card.get_text()
    assert paper['keywords'] in card.get_text() and paper['title'] in card.get_text()
    assert card['data-subcategory']==paper['subcategory']
    assert card.select_one('.translation-trigger')['data-pdf']==paper['pdf_urls'][0]
    assert card.select_one('a.download')['href']==paper['pdf_urls'][0]
    for url in [paper['source'],*paper['pdf_urls'],*paper['code_urls']]:
        parts=urlsplit(url)
        assert parts.scheme in ('http','https') and parts.netloc, (paper['id'],url)
for name in ('hallucination.csv','hallucination_import_report.json'):
    assert (ROOT/'docs'/name).is_file()
exported = list(csv.DictReader((ROOT/'docs/hallucination.csv').open(encoding='utf-8-sig',newline='')))
assert exported==rows
combined = list(csv.DictReader((ROOT/'docs/papers.csv').open(encoding='utf-8-sig',newline='')))
assert len(combined)==len(papers)
report = json.loads((ROOT/'data/hallucination_import_report.json').read_text(encoding='utf-8'))
assert report['paper_count']==79 and report['pdf_count']==79
assert report['categories']==dict(Counter(p['subcategory'] for p in hallucination))
bib=(ROOT/'data/references.bib').read_text(encoding='utf-8')
assert bib.count('@')==len(papers)
for p in hallucination: assert f'eprint = {{{p["arxiv"]}}}' in bib
print('79/79 spreadsheet rows, 9 filters, abstracts, original PDFs, code links, CSV and BibTeX: OK')
