# ocrstruct

`ocrstruct` is a Python library for converting PDF files into:

- markdown text (via MinerU)
- validated `middle.json` structures with page/block/span metadata

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
from ocrstruct import convert_pdf_to_middle, result_to_markdown

result = convert_pdf_to_middle(
    "/path/to/file.pdf",
    outdir="/tmp/ocrstruct-work",
    seal_enable=False,
)

print(result.extracted_by)
print(result_to_markdown(result))
```

## CLI

`pandoc` is optional. If it is installed, `ocrstruct` also writes `text.html`.
Given a PDF, `ocrstruct` writes `middle.json`, `text.md`, and `text.html`.
`text.md` is rendered from the typed `Middle` structure, while `text.html` is rendered from a temporary markdown variant that keeps HTML tables intact.

The same HTML rendering path is also available from the library API via `result_to_html(result)` or `middle_to_html(middle)`.

If [ocrstruct/style.html](/Users/jun/mocrdown/ocrstruct/style.html) exists, it is passed to Pandoc with `--include-in-header`.

```bash
uv run python -m ocrstruct.cli sample.pdf

# Skip seal OCR if your PDFs rarely contain seals/stamps
uv run python -m ocrstruct.cli --disable-seal sample.pdf
```

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

- `convert_pdf_to_middle(pdf_path, *, outdir, backend=None, method=None, lang=None, server_url=None, seal_enable=True, formula_enable=True)`
- `result_to_markdown(result)`
- `middle_to_markdown(middle)`
- `result_to_html(result)`
- `middle_to_html(middle)`
- `markdown_to_html(markdown_text)`
- `Middle`, `Result`, `PageInfo`, `Block`, `Line`, `Span`, `BBox`

## Environment Variables

Used by MinerU backend selection:

- `MINERU_BACKEND` (default: `pipeline`)
- `MINERU_METHOD` (default: `auto`)
- `MINERU_LANG` (default: `japan`)
- `MINERU_SERVER_URL` (optional)

`seal_enable=False` skips MinerU's pipeline seal OCR stage. Layout detection still runs, but the extra OCR pass for detected seal regions is not executed.

## Repository Layout

```text
ocrstruct/
  __init__.py
  middle.py
  middle_to_markdown.py
  pdf.py
  table.py
tests/
pyproject.toml
README.md
```

## Release Checklist

1. Run `pyright` and `pytest`.
2. Bump version in `pyproject.toml`.
3. Create git tag (`vX.Y.Z`).
4. Publish/release.
