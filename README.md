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

## CLI

`pandoc` is optional. If it is installed, `ocrstruct` also writes `text.html`.

If [ocrstruct/style.html](/Users/jun/mocrdown/ocrstruct/style.html) exists, it is passed to Pandoc with `--include-in-header`.

```bash
uv run python -m ocrstruct.cli sample.pdf
```

If `middle.json` already exists, you can skip the expensive OCR step and regenerate `text.md` and `text.html` directly:

```bash
uv run python -m ocrstruct.cli --from-middle /path/to/middle.json
```

When `--from-middle` is used, the output directory defaults to the directory containing `middle.json`.

Example `ocrstruct/style.html`:

```html
<style>
table {
  display: table;
  overflow-x: visible;
  border-collapse: separate;
  border-spacing: 0;
}

thead,
tbody {
  border: 0;
}

th,
td {
  border: 1px solid #cbd5e1;
  padding: 0.5rem 0.75rem;
  vertical-align: top;
}

th {
  background: #f8fafc;
}
</style>
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
