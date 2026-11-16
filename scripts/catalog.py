"""Render catalog records, including the spreadsheet-backed hallucination collection."""
import html
import unicodedata
from collections import Counter
from urllib.parse import urlencode

esc = lambda value: html.escape(str(value or ''), quote=True)
HALLUCINATION_GROUP = '06_视觉幻觉'


def render_card(p, pdf):
    hallucination = p.get('collection') == 'hallucination'
    proposal = p.get('proposal_entry')
    resource = p.get('record_type') == 'resource'
    authors = ', '.join(p['authors'][:3]) + (' et al.' if len(p['authors']) > 3 else '')
    if not authors: authors = '资源仓库' if resource else '作者信息见原始来源'
    search = unicodedata.normalize('NFKC', ' '.join(str(p.get(key,'')) for key in
        ('short','title','title_zh','note','subcategory','keywords','abstract_en','abstract_zh')) + ' ' + ' '.join(p['authors']) +
        (' ' + ' '.join(str(proposal[k]) for k in ('title','category','year_source','relevance')) if proposal else '')).lower()
    meta = f'{p["id"]:02d} · {p.get("published_month") or p.get("year") or ("持续更新" if resource else "年份见来源")}'
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
    proposal_tag = ''
    if proposal:
        proposal_tag = f'<p class="proposal-tag">申博清单 #{proposal["row_id"]:02d} · {esc(proposal["category"])}</p>'
        details += f'<details class="proposal-details"><summary>与申博研究计划的关联</summary><p>{esc(proposal["relevance"])}</p><dl><dt>原表标题</dt><dd>{esc(proposal["title"])}</dd><dt>年份 / 来源（原表）</dt><dd>{esc(proposal["year_source"])}</dd><dt>原表分类</dt><dd>{esc(proposal["category"])}</dd><dt>原表链接</dt><dd><a href="{esc(proposal["link"])}" target="_blank" rel="noopener noreferrer">打开清单中的来源 ↗</a></dd></dl></details>'
    if p.get('pdf_urls'):
        actions=f'<button type="button" class="translation-trigger" data-paper-id="{p["id"]}" data-title="{esc(p["title"])}" data-pdf="{esc(p["pdf_urls"][0])}">翻译阅读</button><a class="download" href="{esc(pdf)}" target="_blank" rel="noopener noreferrer">英文 PDF ↗</a>'
    else:
        actions=''
    source_label = '打开资源仓库' if resource else ('原始来源' if p.get('pdf_urls') else '出版方页面')
    actions += f'<a href="{esc(p["source"])}" target="_blank" rel="noopener noreferrer">{source_label} ↗</a>'+code_links
    return f'''<article id="paper-{p['id']}" data-paper-id="{p['id']}" data-group="{esc(p['group'])}" data-level="{esc(p['level'])}" data-subcategory="{esc(p.get('subcategory',''))}" data-proposal-topic="{esc(proposal['category'] if proposal else '')}" data-record-type="{esc(p.get('record_type','paper'))}" data-search="{esc(search)}">
<div class="meta">{esc(meta)} <span>{esc(badge)}</span></div>{proposal_tag}<h2>{esc(p['short'])}</h2><h3>{esc(p['title'])}</h3>
<p class="authors">{esc(authors)}<br>{esc(p['venue'])}</p><p>{esc(p['note'])}</p>{details}
<footer>{actions}</footer></article>'''


def render_options(papers):
    counts = Counter(p.get('subcategory') for p in papers if p.get('collection') == 'hallucination')
    return ''.join(f'<option value="{esc(topic)}">{esc(topic)}（{count}）</option>' for topic,count in counts.items())


def topic_links(papers):
    counts = Counter(p['subcategory'] for p in papers if p.get('collection') == 'hallucination')
    return '\n'.join(f'| [{topic}](index.html?{urlencode({"group":HALLUCINATION_GROUP,"topic":topic})}) | {count} |' for topic,count in counts.items())


def proposal_members(papers):
    return sorted([p for p in papers if p.get('proposal_entry')],key=lambda p:p['proposal_entry']['row_id'])


def proposal_options(papers):
    counts=Counter(p['proposal_entry']['category'] for p in proposal_members(papers))
    return ''.join(f'<option value="{esc(topic)}">{esc(topic)}（{count}）</option>' for topic,count in counts.items())


def proposal_sections(papers, source):
    members=proposal_members(papers)
    link=lambda p:'index.html?'+urlencode({'group':'07_申博研究计划'})+'#paper-'+str(p['id'])
    entries='\n'.join(f'| {p["proposal_entry"]["row_id"]} | [{p["proposal_entry"]["title"]}]({link(p)}) | {p["proposal_entry"]["category"]} | {p["proposal_entry"]["relevance"]} |' for p in members)
    sheets=list(source['sheets'].values())
    guides='\n'.join(f'| [{row[1]}]({row[2]}) | {row[3]} | {row[4]} |' for row in sheets[1][1:])
    ids={p['proposal_entry']['row_id']:p for p in members}
    import re
    framework=[]
    for row in sheets[2][1:]:
        citations=re.sub(r'\[(\d+)\]',lambda m:f'[{m.group(0)}]({link(ids[int(m.group(1))])})',row[2])
        framework.append(f'| {row[0]} | {row[1]} | {citations} |')
    return {'PROPOSAL_ENTRIES':entries,'PROPOSAL_GUIDES':guides,'PROPOSAL_FRAMEWORK':'\n'.join(framework)}
