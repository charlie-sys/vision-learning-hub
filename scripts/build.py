"""Build the public, portable static site from project sources."""
from pathlib import Path
import argparse
import csv
import html
import json
import re
import shutil

from bs4 import BeautifulSoup
import markdown
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer
from catalog import render_card, render_options, topic_links, proposal_options, proposal_sections

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def colab_url(filename):
    site = json.loads((ROOT / "data/site.json").read_text(encoding="utf-8"))
    return f'https://colab.research.google.com/github/{site["repository"]}/blob/{site["branch"]}/notebooks/{filename}'


def runner_markup(example):
    name, title = html.escape(example["id"], quote=True), html.escape(example["title"])
    source = html.escape(example["code"])
    return f'''<section class="python-runner" data-example="{name}" aria-label="{title}代码运行器">
<div class="runner-heading"><strong>{title}</strong><span>本机浏览器 Python · CPU</span></div>
<label class="runner-editor" for="code-{name}">Python代码（可编辑；Ctrl / Command + Enter 运行）<textarea id="code-{name}" spellcheck="false" autocomplete="off" autocapitalize="off">{source}</textarea></label>
<div class="runner-actions"><button type="button" class="run-code">运行代码</button><button type="button" class="stop-code" disabled>停止</button><button type="button" class="reset-code">恢复示例</button><button type="button" class="download-code">下载代码</button><a href="{colab_url('00_numpy_math.ipynb')}" target="_blank" rel="noopener noreferrer">在 Colab 学习 ↗</a></div>
<p class="runner-status" role="status" aria-live="polite">点击运行后加载Python和所需数学库，首次需要网络。</p>
<pre class="runner-output" aria-label="Python输出"></pre><div class="runner-figures"></div>
<p class="runner-note">每次运行使用独立变量；支持NumPy / SciPy / SymPy / Matplotlib。torch、tensorflow请使用Colab实验。</p></section>'''


def protect_math(text):
    """Keep LaTeX out of Markdown's underscore, escape and table parsing."""
    expressions = []
    pattern = re.compile(r"(```[\s\S]*?```|`[^`\n]*`)|(\$\$[\s\S]*?\$\$|\$[^$\n]+?\$)")

    def replace(match):
        if match.group(1):
            return match.group(0)
        value = match.group(2)
        display = value.startswith("$$")
        tex = value[2:-2].strip() if display else value[1:-1].strip()
        token = f"MATHPLACEHOLDER{len(expressions)}END"
        expressions.append((token, tex, display))
        return "\n\n" + token + "\n\n" if display else token

    return pattern.sub(replace, text), expressions


def render(name, title, code_html="", extra_css=""):
    text = (ROOT / "content" / f"{name}.md").read_text(encoding="utf-8")
    if name == 'hallucination':
        papers = json.loads((ROOT / 'data/papers.json').read_text(encoding='utf-8'))
        text = text.replace('<!-- HALLUCINATION_TOPICS -->', topic_links(papers))
    if name == 'proposal':
        papers=json.loads((ROOT/'data/papers.json').read_text(encoding='utf-8'))
        source=json.loads((ROOT/'data/proposal_source.json').read_text(encoding='utf-8'))
        for key,value in proposal_sections(papers,source).items(): text=text.replace('<!-- '+key+' -->',value)
    examples = {item["id"]: item for item in json.loads((ROOT / "data/runner_examples.json").read_text(encoding="utf-8"))}
    protected, expressions = protect_math(text)
    body = markdown.markdown(protected, extensions=["tables", "fenced_code", "toc", "sane_lists"])
    for token, tex, display in expressions:
        tag = "div" if display else "span"
        kind = "math-display" if display else "math-inline"
        rendered = f'<{tag} class="{kind}" data-tex="{html.escape(tex, quote=True)}">{html.escape(tex)}</{tag}>'
        if display:
            body = body.replace(f"<p>{token}</p>", rendered)
        body = body.replace(token, rendered)
    body = body.replace("<!-- LAB_SOURCE -->", code_html)
    for key, example in examples.items():
        body = body.replace(f"<!-- RUNNER:{key} -->", runner_markup(example))
    for notebook in (ROOT / "notebooks").glob("*.ipynb"):
        body = body.replace(f"<!-- COLAB:{notebook.stem} -->", f'<p class="colab-callout"><a class="colab-link" href="{colab_url(notebook.name)}">在 Colab 运行本实验 ↗</a> · <a href="notebooks/{notebook.name}" download="{notebook.name}">下载Notebook</a> · <a href="playground.html">在线代码实验室</a></p>')
    if "<!-- NOTEBOOKS -->" in body:
        labels = [("00_numpy_math", "数学：NumPy / SciPy / SymPy"), ("01_pytorch_basics", "PyTorch基础"),
                  ("02_tensorflow_basics", "TensorFlow基础"), ("03_cnn", "CNN图像分类"),
                  ("04_vae", "VAE完整实验"), ("05_ddpm", "DDPM完整实验"), ("06_probe", "DINOv2线性评价")]
        entries = ''.join(f'<li><strong>{label}</strong>：<a class="colab-link" href="{colab_url(key+".ipynb")}">在 Colab 运行 ↗</a> · <a href="notebooks/{key}.ipynb" download="{key}.ipynb">下载Notebook</a></li>' for key, label in labels)
        body = body.replace("<!-- NOTEBOOKS -->", '<ul class="notebook-list">'+entries+'</ul>')
    body = re.sub(r"<li>\s*\[ \]", '<li><input type="checkbox" disabled>', body)
    soup = BeautifulSoup(body, "html.parser")
    for a in soup.select("a[href]"):
        if a["href"].startswith("http"):
            a["target"] = "_blank"
            a["rel"] = "noopener noreferrer"
        if name == "tutorial" and a["href"] in ("guide.md", "vision_basics_lab.py"):
            a["href"] = {"guide.md": "guide.html", "vision_basics_lab.py": "lab.html"}[a["href"]]
    items = []
    for h in soup.select("h2,h3"):
        if not h.get("id"):
            h["id"] = f"section-{len(items) + 1}"
        items.append(f'<li class="level-{h.name[1]}"><a href="#{html.escape(h["id"], quote=True)}">{html.escape(h.get_text(" ", strip=True))}</a></li>')
    for table in soup.select("table"):
        labels = [h.get_text(" ", strip=True) for h in table.select("thead th")]
        classes = []
        if labels and labels[0] == "知识点":
            classes.append("resource-table")
        if len(labels) >= 4:
            classes += ["wide-table", f"cols-{len(labels)}"]
        if classes:
            table["class"] = classes
        for row in table.select("tbody tr"):
            for i, cell in enumerate(row.find_all("td", recursive=False)):
                cell["data-label"] = labels[i] if i < len(labels) else ""
    body = str(soup) + f'<p><a href="{name}.md" download="{name}.md">下载本页 Markdown 源文</a></p>'
    template = (ROOT / "templates" / "page.html").read_text(encoding="utf-8")
    page = template.replace("{{title}}", html.escape(title)).replace("{{toc}}", '<div class="toc"><ul>' + ''.join(items) + '</ul></div>').replace("{{body}}", body).replace("{{extra_css}}", extra_css)
    (DOCS / f"{name}.html").write_text(page, encoding="utf-8")
    shutil.copy2(ROOT / "content" / f"{name}.md", DOCS / f"{name}.md")


def build(local_pdfs=False):
    DOCS.mkdir(exist_ok=True)
    shutil.copytree(ROOT / "vendor" / "katex", DOCS / "assets" / "katex", dirs_exist_ok=True)
    shutil.copy2(ROOT / "assets" / "math.js", DOCS / "assets" / "math.js")
    shutil.copy2(ROOT / "assets" / "translation.js", DOCS / "assets" / "translation.js")
    render("tutorial", "视觉论文前置知识完整教程")
    render("guide", "视觉论文完整指南")
    render("translation", "英文论文翻译阅读指南")
    for name, title in [("playground", "在线代码实验室"), ("math", "数学推导与数值例题"), ("practice", "四项完整视觉实验"),
                        ("answers", "练习答案与纠错"), ("resources", "可靠课程与知识映射"),
                        ("assessment", "学习路线与考核标准"), ("evidence", "实验实测记录"),
                        ("hallucination", "视觉幻觉论文专题"), ("proposal", "申博研究计划书文献与资源")]:
        render(name, title)
    for asset in ("runner.js", "runner-worker.js", "runner.css", "catalog.js"):
        shutil.copy2(ROOT / "assets" / asset, DOCS / "assets" / asset)
    shutil.copytree(ROOT / "notebooks", DOCS / "notebooks", dirs_exist_ok=True)
    for name in ("course_labs.py", "solutions.py", "exercises.py"):
        shutil.copy2(ROOT / "examples" / name, DOCS / name)
    if (ROOT / "evidence").exists():
        shutil.copytree(ROOT / "evidence", DOCS / "evidence", dirs_exist_ok=True)
    formatter = HtmlFormatter(style="monokai", linenos="inline", cssclass="highlight")
    code = (ROOT / "examples" / "vision_basics_lab.py").read_text(encoding="utf-8")
    css = formatter.get_style_defs(".highlight") + "\n.highlight{overflow:auto;border-radius:9px}.highlight pre{min-width:max-content;margin:0}.highlight .linenos{color:#91a0a8;margin-right:12px;user-select:none}"
    render("lab", "PyTorch 视觉基础小实验", highlight(code, PythonLexer(), formatter), css)
    shutil.copy2(ROOT / "examples" / "vision_basics_lab.py", DOCS / "vision_basics_lab.py")
    papers = json.loads((ROOT / "data" / "papers.json").read_text(encoding="utf-8"))
    cards = []
    for p in papers:
        pdf = (p.get('pdf_urls') or [p['source']])[0]
        if local_pdfs and (DOCS / "PDFs" / p["filename"]).is_file():
            pdf = "PDFs/" + p["filename"]
        cards.append(render_card(p, pdf))
    catalog = (ROOT / "templates" / "catalog.html").read_text(encoding="utf-8").replace("{{cards}}", ''.join(cards))
    replacements = {'paper_count':sum(p.get('record_type')!='resource' for p in papers), 'entry_count':len(papers),
                    'resource_count':sum(p.get('record_type')=='resource' for p in papers),'topic_count':len({p['group'] for p in papers}),
                    'hallucination_count':sum(p.get('collection')=='hallucination' for p in papers),
                    'hallucination_options':render_options(papers), 'proposal_options':proposal_options(papers), 'catalog_date':'2026-11-16'}
    for key,value in replacements.items():
        catalog = catalog.replace('{{'+key+'}}',str(value))
    (DOCS / "index.html").write_text(catalog, encoding="utf-8")
    with (DOCS / "papers.csv").open("w", encoding="utf-8-sig", newline="") as f:
        fields = ["id", "record_type", "group", "subcategory", "spreadsheet_id", "short", "title", "title_zh", "authors", "year", "venue", "level", "note", "keywords", "source", "pdf_url", "proposal_row", "proposal_category", "proposal_relevance"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for p in papers:
            row = {key: p.get(key, "") for key in fields}
            row["authors"] = '; '.join(p["authors"])
            row["pdf_url"] = p['pdf_urls'][0] if p.get('pdf_urls') else ''
            row['record_type']=p.get('record_type','paper')
            if p.get('proposal_entry'):
                row.update(proposal_row=p['proposal_entry']['row_id'],proposal_category=p['proposal_entry']['category'],proposal_relevance=p['proposal_entry']['relevance'])
            writer.writerow(row)
    shutil.copy2(ROOT / "data" / "references.bib", DOCS / "references.bib")
    shutil.copy2(ROOT / "data" / "papers.json", DOCS / "papers.json")
    shutil.copy2(ROOT / 'data/hallucination_source.csv', DOCS / 'hallucination.csv')
    shutil.copy2(ROOT / 'data/hallucination_import_report.json', DOCS / 'hallucination_import_report.json')
    for name in ('proposal_source.csv','proposal_source.json','proposal_import_report.json'):
        shutil.copy2(ROOT/'data'/name,DOCS/name)
    (DOCS / ".nojekyll").touch()
    print(f"Built {len(list(DOCS.glob('*.html')))} HTML pages and {len(papers)} catalog entries in docs/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-pdfs", action="store_true", help="Link locally downloaded PDFs where present")
    build(parser.parse_args().local_pdfs)
