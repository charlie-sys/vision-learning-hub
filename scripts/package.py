"""Package the reusable project, excluding Git internals and local paper PDFs."""
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
output = ROOT / "dist" / "vision-learning-hub.zip"
output.parent.mkdir(exist_ok=True)
with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as z:
    for path in ROOT.rglob("*"):
        rel = path.relative_to(ROOT)
        if not path.is_file() or path.suffix in (".pt", ".pth") or any(part in (".git", ".venv", "__pycache__", "dist", "PDFs", "runs", "datasets", ".cache") for part in rel.parts):
            continue
        z.write(path, Path(ROOT.name) / rel)
with zipfile.ZipFile(output) as z:
    assert z.testzip() is None
print(output)
