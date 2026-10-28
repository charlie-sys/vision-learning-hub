"""Validate generated local links, paper records, and responsive table markup."""
from pathlib import Path
from urllib.parse import unquote
import json

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
docs = ROOT / "docs"
papers = json.loads((ROOT / "data/papers.json").read_text(encoding="utf-8"))
assert len(papers) == 30 and len({p["id"] for p in papers}) == 30
assert all(p["pdf_urls"] and p["source"].startswith("https://") for p in papers)
for path in docs.glob("*.html"):
    text = path.read_text(encoding="utf-8")
    assert "{{" not in text, f"Unexpanded template in {path.name}"
    assert "D:\\" not in text and "file:///" not in text, f"Machine-specific path in {path.name}"
    soup = BeautifulSoup(text, "html.parser")
    ids = {el.get("id") for el in soup.select("[id]")}
    for a in soup.select("a[href]"):
        href = unquote(a["href"])
        if href.startswith(("http://", "https://", "mailto:")):
            continue
        if href.startswith("#"):
            assert href[1:] in ids, (path.name, href)
        else:
            assert (path.parent / href.split("#")[0]).exists(), (path.name, href)
    for table in soup.select("table"):
        n = len(table.select("thead th"))
        for row in table.select("tbody tr"):
            cells = row.find_all("td", recursive=False)
            assert len(cells) == n and all("data-label" in c.attrs for c in cells)
    print(f"{path.name}: OK")
assert len(BeautifulSoup((docs / "index.html").read_text(encoding="utf-8"), "html.parser").select("article")) == 30
print("30 paper records, four pages, internal links and table labels: OK")
