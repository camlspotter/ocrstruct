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
from ocrstruct.api import (
    convert_pdf_to_middle,
    merge_discarded_blocks,
    result_to_markdown,
)

result = convert_pdf_to_middle(
    "/path/to/file.pdf",
    outdir="/tmp/ocrstruct-work",
    seal_enable=False,
)
result.middle_json = merge_discarded_blocks(result.middle_json)

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

# Generate image screening JSONL directly from middle.json
uv run ocrstruct-screening \
  --middle-json /tmp/ocrstruct-work/middle.json \
  --out /tmp/ocrstruct-work/image_screening.jsonl \
  --model gpt-5.4-mini

# Generate understanding JSONL from prior screening results
uv run ocrstruct-understanding \
  --screening-results screening/screening_result_for_understanding.jsonl \
  --out understanding/image_understanding_results.jsonl \
  --model gpt-5.4-mini \
  --skip-existing
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

Recommended external API (from `ocrstruct.api`):

- `convert_pdf_to_middle(pdf_path, *, outdir, backend=None, method=None, lang=None, server_url=None, seal_enable=True, formula_enable=True, lazy=False)`
- `merge_discarded_blocks(middle)`
- `result_to_markdown(result)`
- `middle_to_markdown(middle)`
- `result_to_html(result)`
- `middle_to_html(middle)`
- `markdown_to_html(markdown_text)`
- `image_refs_from_middle(middle, *, pdf_path, middle_json_path)`
- `load_image_refs_from_middle_json(middle_json_path, *, pdf_path=None)`
- `iter_screening_records_from_refs(refs, *, model, pricing, base_url=None, api_key=None, thinking=False, existing_keys=None)`
- `load_screening_records_jsonl(path, *, screening_thinking=None)`
- `iter_understanding_records_from_screening(screening_records, *, model, pricing, base_url=None, api_key=None, thinking=False, existing_keys=None)`
- `image_understanding_run_from_screening(ref, screening, *, model, base_url=None, api_key=None, pricing, thinking=False)`
- `pricing_for_model(model, pricing_overrides=None)`
- `load_pricing_overrides(path)`
- `build_images_file(records, *, middle_json_path, generated_at=None)`
- `load_images_file_json(path, *, middle_json_sha256=None, middle_json_path=None)`
- `merge_understanding_into_middle(middle, records)`
- `merge_images_into_middle(middle, images_file)`
- `compute_middle_json_sha256(path)`

The package root `ocrstruct` still re-exports many symbols for compatibility, but `ocrstruct.api` is the intended stable function-level surface.

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
