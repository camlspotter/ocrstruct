# ocrstruct

`ocrstruct` is a Python library for converting PDF files into:

- markdown text (via MinerU)
- structured elements (text/title/image/table/code/math with bbox/page metadata)

## Install

### From Git

```bash
pip install "git+https://github.com/<your-org>/ocrstruct.git@v0.1.0"
```

### Editable (local development)

```bash
pip install -e .[dev]
```

## Quick Start

```python
from ocrstruct import convert_pdf_to_elements

elements = convert_pdf_to_elements(
    "/path/to/file.pdf",
    tmpdir="/tmp/ocrstruct-work",
)

for e in elements[:5]:
    print(e.kind, e.text)
```

## API

Public API (from `ocrstruct.__init__`):

- `convert_pdf_to_elements(pdf_path, *, tmpdir, img_bucket_path="images", backend=None, method=None, lang=None, server_url=None)`
- `Element`, `Location`, `BBox`

## Environment Variables

Used by MinerU backend selection:

- `MINERU_BACKEND` (default: `pipeline`)
- `MINERU_METHOD` (default: `auto`)
- `MINERU_LANG` (default: `japan`)
- `MINERU_SERVER_URL` (optional)

## Repository Layout

```text
ocrstruct/
  __init__.py
  pdf.py
  middle_to_elements.py
  table.py
  types.py
tests/
pyproject.toml
README.md
```

## Release Checklist

1. Run `pyright` and `pytest`.
2. Bump version in `pyproject.toml`.
3. Create git tag (`vX.Y.Z`).
4. Publish/release.
