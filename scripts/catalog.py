"""Render catalog records, including the spreadsheet-backed hallucination collection."""
import html
import unicodedata
from collections import Counter
from urllib.parse import urlencode

esc = lambda value: html.escape(str(value or ''), quote=True)
HALLUCINATION_GROUP = '06_视觉幻觉'


def render_card(p, pdf):
    hallucination = p.get('collection') == 'hallucination'
    authors = ', '.join(p['authors'][:3]) + (' et al.' if len(p['authors']) > 3 else '')
    search = unicodedata.normalize('NFKC', ' '.join(str(p.get(key,'')) for key in
        ('short','title','title_zh','note','subcategory','keywords','abstract_en','abstract_zh')) + ' ' + ' '.join(p['authors'])).lower()
    meta = f'{p["id"]:02d} · {p.get("published_month", p["year"])}'
    if hallucination: meta += f' · 原表 #{p["spreadsheet_id"]:02d}'
    badge = p['subcategory'] if hallucination else p['level']
    details = ''
    if hallucination:
        fields = [('完整作者', ', '.join(p['authors'])),('发表时间',p['published_month']),('原表序号',p['spreadsheet_id']),
                  ('研究方向',p['subcategory']),('关键词',p['keywords']),('发表载体类型',p['publication_type']),
                  ('正式发表载体',p['venue']),('CCF推荐等级',p['ccf']),('中科院分区',p['cas_quartile']),
                  ('JCR分区',p['jcr_quartile']),('影响因子',p['impact_factor']),('分区与收录说明',p['publication_notes']),
                  ('arXiv录用备注',p['arxiv_acceptance_note'] or '原表未注明'),('arXiv主分类',p['arxiv_category'])]
        details = '<details class="paper-details"><summary>展开中英文摘要与表格信息</summary>'
        details += f'<h4>中文摘要（原表译文）</h4><p>{esc(p["abstract_zh"])}</p><h4>English abstract</h4><p lang="en">{esc(p["abstract_en"])}</p>'
        details += '<dl>'+''.join(f'<dt>{esc(k)}</dt><dd>{esc(v)}</dd>' for k,v in fields)+'</dl><p class="provenance">发表与分区信息按原表记录；中文译名与译文用于辅助阅读，正式引用请使用英文原题。</p></details>'
    code_links = ''.join(f'<a href="{esc(url)}" target="_blank" rel="noopener noreferrer">代码/项目{index if len(p.get("code_urls",[]))>1 else ""} ↗</a>' for index,url in enumerate(p.get('code_urls',[]),1))
    return f'''<article id="paper-{p['id']}" data-paper-id="{p['id']}" data-group="{esc(p['group'])}" data-level="{esc(p['level'])}" data-subcategory="{esc(p.get('subcategory',''))}" data-search="{esc(search)}">
<div class="meta">{esc(meta)} <span>{esc(badge)}</span></div><h2>{esc(p['short'])}</h2><h3>{esc(p['title'])}</h3>
<p class="authors">{esc(authors)}<br>{esc(p['venue'])}</p><p>{esc(p['note'])}</p>{details}
<footer><button type="button" class="translation-trigger" data-paper-id="{p['id']}" data-title="{esc(p['title'])}" data-pdf="{esc(p['pdf_urls'][0])}">翻译阅读</button><a class="download" href="{esc(pdf)}" target="_blank" rel="noopener noreferrer">英文 PDF ↗</a><a href="{esc(p['source'])}" target="_blank" rel="noopener noreferrer">原始来源 ↗</a>{code_links}</footer></article>'''


def render_options(papers):
    counts = Counter(p.get('subcategory') for p in papers if p.get('collection') == 'hallucination')
    return ''.join(f'<option value="{esc(topic)}">{esc(topic)}（{count}）</option>' for topic,count in counts.items())


def topic_links(papers):
    counts = Counter(p['subcategory'] for p in papers if p.get('collection') == 'hallucination')
    return '\n'.join(f'| [{topic}](index.html?{urlencode({"group":HALLUCINATION_GROUP,"topic":topic})}) | {count} |' for topic,count in counts.items())
