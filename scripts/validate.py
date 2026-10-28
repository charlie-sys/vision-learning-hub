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
    assert len(ids) == len(soup.select("[id]")), f"Duplicate anchors in {path.name}"
    assert "MATHPLACEHOLDER" not in text, f"Unrestored equation in {path.name}"
    for a in soup.select("a[href]"):
        href = unquote(a["href"])
        if href.startswith(("http://", "https://", "mailto:")):
            continue
        if href.startswith("#"):
            assert href[1:] in ids, (path.name, href)
        else:
            assert (path.parent / href.split("#")[0]).exists(), (path.name, href)
            if "#" in href:
                target, anchor = href.split("#", 1)
                other = BeautifulSoup((path.parent / target).read_text(encoding="utf-8"), "html.parser")
                assert other.find(id=anchor), (path.name, href)
    for img in soup.select("img[src]"):
        if not img["src"].startswith(("http://", "https://")):
            assert (path.parent / unquote(img["src"])).is_file(), (path.name, img["src"])
    if path.name != "index.html":
        assert soup.select_one(f'a[download="{path.stem}.md"]'), f"Missing source download in {path.name}"
    for asset in soup.select("script[src],link[rel=stylesheet][href]"):
        url = asset.get("src") or asset["href"]
        if not url.startswith(("http://", "https://")):
            assert (path.parent / url).is_file(), (path.name, url)
    for table in soup.select("table"):
        n = len(table.select("thead th"))
        for row in table.select("tbody tr"):
            cells = row.find_all("td", recursive=False)
            assert len(cells) == n and all("data-label" in c.attrs for c in cells)
    print(f"{path.name}: OK")
assert len(BeautifulSoup((docs / "index.html").read_text(encoding="utf-8"), "html.parser").select("article")) == 30
catalog = BeautifulSoup((docs / "index.html").read_text(encoding="utf-8"), "html.parser")
translation_buttons = catalog.select("button.translation-trigger")
assert len(translation_buttons) == 30
expected_pdfs = {str(p["id"]): p["pdf_urls"][0] for p in papers}
assert all(button["data-pdf"] == expected_pdfs[button["data-paper-id"]] for button in translation_buttons)
assert catalog.select_one("dialog#translation-dialog") is not None
math_page = BeautifulSoup((docs / "tutorial.html").read_text(encoding="utf-8"), "html.parser")
assert len(math_page.select("[data-tex]")) >= 40, "Mathematical expressions were not preserved"
assert len(math_page.select(".math-display")) >= 10, "Missing display equations"
assert "MATHPLACEHOLDER" not in str(math_page), "Unrestored equation placeholder"
print("30 paper records, all generated pages, translation entries, internal links and table labels: OK")
