"""Download public paper PDFs directly from the recorded original sources."""
from pathlib import Path
import argparse
import json
import time

import requests

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", help="Comma-separated paper IDs, e.g. 1,2,15")
    parser.add_argument("--list", action="store_true", help="List papers without downloading")
    args = parser.parse_args()
    papers = json.loads((ROOT / "data/papers.json").read_text(encoding="utf-8"))
    if args.ids:
        selected = {int(value) for value in args.ids.split(",")}
        known = {p["id"] for p in papers}
        if selected - known:
            parser.error(f"Unknown paper IDs: {sorted(selected - known)}")
        papers = [p for p in papers if p["id"] in selected]
    if args.list:
        for p in papers:
            print(f'{p["id"]:02d} {p["short"]}: {p["title"]}')
        return
    output = ROOT / "docs/PDFs"
    output.mkdir(parents=True, exist_ok=True)
    client = requests.Session()
    client.headers["User-Agent"] = "vision-learning-hub/1.0 (public academic sources)"
    failed = []
    for p in papers:
        target = output / p["filename"]
        if target.is_file() and target.stat().st_size > 1000:
            with target.open("rb") as f:
                if f.read(5) == b"%PDF-":
                    print(f'{p["id"]:02d} already downloaded')
                    continue
        part = target.with_suffix(".part")
        success = False
        for url in p["pdf_urls"]:
            try:
                print(f'{p["id"]:02d} downloading {p["short"]}', flush=True)
                with client.get(url, stream=True, timeout=(20, 60)) as response:
                    response.raise_for_status()
                    with part.open("wb") as f:
                        for chunk in response.iter_content(256 * 1024):
                            f.write(chunk)
                with part.open("rb") as f:
                    if f.read(5) != b"%PDF-":
                        raise ValueError("Source returned non-PDF content")
                part.replace(target)
                success = True
                print(f'{p["id"]:02d} saved {target.name}')
                break
            except (requests.RequestException, OSError, ValueError) as error:
                print(f"  Source failed: {type(error).__name__}")
        if not success:
            failed.append(p)
        time.sleep(1)
    if failed:
        print("Could not download:")
        for p in failed:
            print(f'{p["id"]:02d} {p["title"]}: {p["source"]}')
        raise SystemExit(1)
    print("Downloads complete. Run: python scripts/build.py --local-pdfs")


if __name__ == "__main__":
    main()
