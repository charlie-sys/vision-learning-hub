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

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


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
    body = str(soup) + f'<p><a href="{name}.md">下载本页 Markdown 源文</a></p>'
    template = (ROOT / "templates" / "page.html").read_text(encoding="utf-8")
    page = template.replace("{{title}}", html.escape(title)).replace("{{toc}}", '<div class="toc"><ul>' + ''.join(items) + '</ul></div>').replace("{{body}}", body).replace("{{extra_css}}", extra_css)
    (DOCS / f"{name}.html").write_text(page, encoding="utf-8")
    shutil.copy2(ROOT / "content" / f"{name}.md", DOCS / f"{name}.md")


def build(local_pdfs=False):
    DOCS.mkdir(exist_ok=True)
    shutil.copytree(ROOT / "vendor" / "katex", DOCS / "assets" / "katex", dirs_exist_ok=True)
    shutil.copy2(ROOT / "assets" / "math.js", DOCS / "assets" / "math.js")
    render("tutorial", "视觉论文前置知识完整教程")
    render("guide", "视觉论文完整指南")
    formatter = HtmlFormatter(style="monokai", linenos="inline", cssclass="highlight")
    code = (ROOT / "examples" / "vision_basics_lab.py").read_text(encoding="utf-8")
    css = formatter.get_style_defs(".highlight") + "\n.highlight{overflow:auto;border-radius:9px}.highlight pre{min-width:max-content;margin:0}.highlight .linenos{color:#91a0a8;margin-right:12px;user-select:none}"
    render("lab", "PyTorch 视觉基础小实验", highlight(code, PythonLexer(), formatter), css)
    shutil.copy2(ROOT / "examples" / "vision_basics_lab.py", DOCS / "vision_basics_lab.py")
    papers = json.loads((ROOT / "data" / "papers.json").read_text(encoding="utf-8"))
    cards = []
    esc = lambda value: html.escape(str(value), quote=True)
    for p in papers:
        pdf = p["pdf_urls"][0]
        if local_pdfs and (DOCS / "PDFs" / p["filename"]).is_file():
            pdf = "PDFs/" + p["filename"]
        authors = ', '.join(p["authors"][:3]) + (' et al.' if len(p["authors"]) > 3 else '')
        haystack = ' '.join([p["short"], p["title"], p["note"], ' '.join(p["authors"])]).lower()
        cards.append(f'<article data-group="{esc(p["group"])}" data-level="{esc(p["level"])}" data-search="{esc(haystack)}"><div class="meta">{p["id"]:02d} · {p["year"]} <span>{esc(p["level"])}</span></div><h2>{esc(p["short"])}</h2><h3>{esc(p["title"])}</h3><p class="authors">{esc(authors)}<br>{esc(p["venue"])}</p><p>{esc(p["note"])}</p><footer><a class="download" href="{esc(pdf)}" target="_blank" rel="noopener noreferrer">打开论文 PDF ↗</a><a href="{esc(p["source"])}" target="_blank" rel="noopener noreferrer">原始来源 ↗</a></footer></article>')
    catalog = (ROOT / "templates" / "catalog.html").read_text(encoding="utf-8").replace("{{cards}}", ''.join(cards))
    (DOCS / "index.html").write_text(catalog, encoding="utf-8")
    with (DOCS / "papers.csv").open("w", encoding="utf-8-sig", newline="") as f:
        fields = ["id", "group", "short", "title", "authors", "year", "venue", "level", "note", "source", "pdf_url"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for p in papers:
            row = {key: p.get(key, "") for key in fields}
            row["authors"] = '; '.join(p["authors"])
            row["pdf_url"] = p["pdf_urls"][0]
            writer.writerow(row)
    shutil.copy2(ROOT / "data" / "references.bib", DOCS / "references.bib")
    shutil.copy2(ROOT / "data" / "papers.json", DOCS / "papers.json")
    (DOCS / ".nojekyll").touch()
    print(f"Built four HTML pages and {len(papers)} paper records in docs/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-pdfs", action="store_true", help="Link locally downloaded PDFs where present")
    build(parser.parse_args().local_pdfs)
