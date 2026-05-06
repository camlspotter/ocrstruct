from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from ocrstruct.image_understanding import ImageRef
from scripts.run_image_screening_eval import EvalRecord


KIND_OPTIONS = (
    "diagram",
    "table_or_form",
    "chart_or_graph",
    "ui_or_screenshot",
    "arrow_only",
    "code_symbol",
    "seal",
    "text_as_image",
    "decorative",
    "logo_or_mark",
    "other",
)

RAG_OPTIONS = (
    "high",
    "medium",
    "low",
    "none",
)

DETAIL_OPTIONS = (
    "skip",
    "short",
    "long",
    "extract_text",
)


class GoldsetSeedItem(dict[str, str]):
    pass


def _load_records(path: Path) -> list[EvalRecord]:
    return [
        EvalRecord.model_validate_json(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def _load_seed_items(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Seed JSON must be an object: {path}")
    items = data.get("items")
    if not isinstance(items, list):
        raise ValueError(f"Seed JSON must contain an items list: {path}")
    out: dict[str, dict[str, str]] = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError(f"Seed item must be an object: {path}")
        item_key = item.get("item_key")
        if not isinstance(item_key, str):
            raise ValueError(f"Seed item must contain item_key: {path}")
        out[item_key] = {
            key: value
            for key, value in item.items()
            if isinstance(key, str) and isinstance(value, str)
        }
    return out


def _image_src(ref: ImageRef) -> str:
    image_path = Path(ref.image_path)
    if image_path.is_absolute():
        return str(image_path)
    return str(Path(ref.middle_json_path).parent / "images" / image_path)


def _pdf_href(ref: ImageRef) -> str:
    return f"{ref.pdf_path}#page={ref.page_idx + 1}"


def _text_block(value: str | None) -> str:
    if value is None:
        return ""
    return html.escape(value).replace("\n", "<br>")


def _record_key(ref: ImageRef) -> str:
    return "::".join(
        [
            ref.middle_json_path,
            str(ref.page_idx),
            str(ref.block_index),
            ref.image_path,
        ]
    )


def _radio_group(name: str, options: tuple[str, ...], selected: str) -> str:
    parts: list[str] = ['<div class="radio-group">']
    for option in options:
        checked = ' checked' if option == selected else ""
        option_id = f"{name}-{option}"
        parts.append(
            '<label class="radio-option" for="'
            + html.escape(option_id)
            + '">'
            + '<input type="radio" id="'
            + html.escape(option_id)
            + '" name="'
            + html.escape(name)
            + '" value="'
            + html.escape(option)
            + '"'
            + checked
            + ">"
            + "<span>"
            + html.escape(option)
            + "</span>"
            + "</label>"
        )
    parts.append("</div>")
    return "".join(parts)


def _default_output_path(input_path: Path, model: str) -> Path:
    return input_path.with_name(f"{input_path.stem}.{model}.goldset.html")


def _render_item(
    index: int,
    record: EvalRecord,
    model: str,
    seed_item: dict[str, str] | None,
) -> str:
    if record.run is None:
        raise ValueError("Expected successful run record")

    ref = record.ref
    item_id = f"item-{index}"
    key = _record_key(ref)
    kind = seed_item.get("kind", record.run.kind) if seed_item is not None else record.run.kind
    rag_value = (
        seed_item.get("rag_value", record.run.rag_value)
        if seed_item is not None
        else record.run.rag_value
    )
    detail_level = (
        seed_item.get("detail_level", record.run.detail_level)
        if seed_item is not None
        else record.run.detail_level
    )
    notes = seed_item.get("notes", record.run.notes or "") if seed_item is not None else (record.run.notes or "")
    return f"""
  <section class="item" data-item-key="{html.escape(key)}">
    <h2>{index + 1}. {html.escape(ref.image_path)}</h2>
    <div class="meta">
      <div><strong>model seed:</strong> {html.escape(model)}</div>
      <div><strong>pdf_path:</strong> <a href="{html.escape(_pdf_href(ref), quote=True)}" target="_blank" rel="noopener noreferrer">{html.escape(ref.pdf_path)}</a></div>
      <div><strong>middle_json_path:</strong> <a href="{html.escape(ref.middle_json_path, quote=True)}" target="_blank" rel="noopener noreferrer">{html.escape(ref.middle_json_path)}</a></div>
      <div><strong>page_idx:</strong> {ref.page_idx}</div>
      <div><strong>block_index:</strong> {ref.block_index}</div>
      <div><strong>block_type:</strong> {html.escape(ref.block_type)}</div>
    </div>
    <img src="{html.escape(_image_src(ref), quote=True)}" alt="{html.escape(ref.image_path)}">
    <div class="annotation">
      <div class="field">
        <div class="field-label">kind</div>
        {_radio_group(f"{item_id}-kind", KIND_OPTIONS, kind)}
      </div>
      <div class="field">
        <div class="field-label">rag_value</div>
        {_radio_group(f"{item_id}-rag", RAG_OPTIONS, rag_value)}
      </div>
      <div class="field">
        <div class="field-label">detail_level</div>
        {_radio_group(f"{item_id}-detail", DETAIL_OPTIONS, detail_level)}
      </div>
      <div class="field">
        <label class="field-label" for="{item_id}-notes">notes</label>
        <textarea id="{item_id}-notes" rows="2">{html.escape(notes)}</textarea>
      </div>
    </div>
    <div class="context-stack">
      <div class="context-box"><strong>caption</strong><div>{_text_block(ref.caption)}</div></div>
      <div class="context-box"><strong>section_title</strong><div>{_text_block(ref.section_title)}</div></div>
      <div class="context-box"><strong>nearby_text_before</strong><div>{_text_block(ref.nearby_text_before)}</div></div>
      <div class="context-box"><strong>nearby_text_after</strong><div>{_text_block(ref.nearby_text_after)}</div></div>
    </div>
    <details>
      <summary>seed response</summary>
      <pre>{html.escape(record.run.raw_text)}</pre>
    </details>
  </section>
"""


def _render_html(
    records: list[EvalRecord],
    model: str,
    input_path: Path,
    seed_json_path: Path | None,
    seed_items: dict[str, dict[str, str]],
) -> str:
    items = "".join(
        _render_item(index, record, model, seed_items.get(_record_key(record.ref)))
        for index, record in enumerate(records)
    )
    export_name = f"{input_path.stem}.{model}.goldset.json"
    storage_key = f"goldset:{input_path.resolve()}:{model}:{seed_json_path.resolve() if seed_json_path is not None else 'none'}"
    return f"""<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>gold set annotation: {html.escape(model)}</title>
  <style>
    :root {{
      --border: #d9d9d9;
      --muted: #666;
      --bg: #fafafa;
      --accent: #0f766e;
    }}
    body {{
      font-family: Helvetica, Arial, sans-serif;
      line-height: 1.5;
      margin: 24px auto;
      max-width: 1100px;
      padding: 0 20px 48px;
      color: #222;
      background: white;
    }}
    .toolbar {{
      position: sticky;
      top: 0;
      background: rgba(255, 255, 255, 0.96);
      backdrop-filter: blur(6px);
      border-bottom: 1px solid var(--border);
      padding: 12px 0;
      margin-bottom: 20px;
    }}
    .toolbar-inner {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: center;
    }}
    button {{
      border: 1px solid var(--border);
      background: white;
      padding: 8px 12px;
      border-radius: 8px;
      cursor: pointer;
    }}
    .status {{
      color: var(--muted);
      font-size: 14px;
    }}
    .item {{
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 14px;
      margin-bottom: 18px;
      background: var(--bg);
    }}
    h1, h2 {{
      line-height: 1.2;
    }}
    img {{
      display: block;
      max-width: 50vw;
      height: auto;
      margin: 8px 0 10px;
      border: 1px solid var(--border);
      background: white;
    }}
    .meta {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 4px 12px;
      font-size: 13px;
      margin-bottom: 8px;
    }}
    .context-stack {{
      display: grid;
      gap: 8px;
      margin-top: 8px;
      margin-bottom: 10px;
    }}
    .context-box {{
      border: 1px solid var(--border);
      border-radius: 6px;
      background: white;
      padding: 8px 10px;
      font-size: 13px;
      min-height: 0;
    }}
    .context-box strong {{
      display: block;
      margin-bottom: 4px;
    }}
    .annotation {{
      display: grid;
      gap: 10px;
    }}
    .field {{
      background: white;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 8px 10px;
    }}
    .field-label {{
      display: block;
      font-weight: 700;
      margin-bottom: 6px;
    }}
    .radio-group {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }}
    .radio-option {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      border: 1px solid var(--border);
      border-radius: 999px;
      padding: 4px 8px;
      background: #fff;
      cursor: pointer;
      font-size: 13px;
    }}
    textarea {{
      width: 100%;
      box-sizing: border-box;
      font: inherit;
      padding: 6px 8px;
    }}
    pre {{
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      background: white;
      padding: 8px;
      border: 1px solid var(--border);
      border-radius: 6px;
      font-size: 13px;
    }}
    a {{
      color: var(--accent);
      word-break: break-all;
    }}
  </style>
</head>
<body>
  <div class="toolbar">
    <div class="toolbar-inner">
      <strong>gold set annotation</strong>
      <span>seed model: {html.escape(model)}</span>
      <span>items: {len(records)}</span>
      <button type="button" id="save-draft">下書きを保存</button>
      <button type="button" id="export-json">JSON を書き出す</button>
      <button type="button" id="clear-draft">下書きを消す</button>
      <span class="status" id="status">自動保存は localStorage に入ります</span>
    </div>
  </div>

  <h1>gold set annotation</h1>
  <p>元ファイル: {html.escape(str(input_path))}</p>
  <p>`gpt-5.4` の回答を初期値として読み込み、radio button で正解ラベルを選ぶ想定です。</p>

{items}

  <script>
    const storageKey = {json.dumps(storage_key)};
    const exportName = {json.dumps(export_name)};

    function collectItems() {{
      return Array.from(document.querySelectorAll('.item')).map((section, index) => {{
        const itemKey = section.dataset.itemKey;
        const kind = section.querySelector('input[name="item-' + index + '-kind"]:checked')?.value ?? null;
        const ragValue = section.querySelector('input[name="item-' + index + '-rag"]:checked')?.value ?? null;
        const detailLevel = section.querySelector('input[name="item-' + index + '-detail"]:checked')?.value ?? null;
        const notes = section.querySelector('#item-' + index + '-notes')?.value ?? '';
        return {{
          item_key: itemKey,
          kind: kind,
          rag_value: ragValue,
          detail_level: detailLevel,
          notes: notes,
        }};
      }});
    }}

    function saveDraft() {{
      const payload = collectItems();
      window.localStorage.setItem(storageKey, JSON.stringify(payload));
      document.getElementById('status').textContent = '下書きを保存しました';
    }}

    function loadDraft() {{
      const raw = window.localStorage.getItem(storageKey);
      if (raw === null) {{
        return;
      }}
      const payload = JSON.parse(raw);
      payload.forEach((item, index) => {{
        const kind = document.querySelector('input[name="item-' + index + '-kind"][value="' + item.kind + '"]');
        const rag = document.querySelector('input[name="item-' + index + '-rag"][value="' + item.rag_value + '"]');
        const detail = document.querySelector('input[name="item-' + index + '-detail"][value="' + item.detail_level + '"]');
        const notes = document.querySelector('#item-' + index + '-notes');
        if (kind) kind.checked = true;
        if (rag) rag.checked = true;
        if (detail) detail.checked = true;
        if (notes) notes.value = item.notes ?? '';
      }});
      document.getElementById('status').textContent = '保存済み下書きを読み込みました';
    }}

    function clearDraft() {{
      window.localStorage.removeItem(storageKey);
      document.getElementById('status').textContent = '保存済み下書きを消しました。ページ再読込で初期値に戻ります';
    }}

    function exportJson() {{
      const payload = {{
        seed_model: {json.dumps(model)},
        items: collectItems(),
      }};
      const blob = new Blob([JSON.stringify(payload, null, 2)], {{ type: 'application/json' }});
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = exportName;
      a.click();
      URL.revokeObjectURL(url);
      document.getElementById('status').textContent = 'JSON を書き出しました';
    }}

    document.getElementById('save-draft').addEventListener('click', saveDraft);
    document.getElementById('export-json').addEventListener('click', exportJson);
    document.getElementById('clear-draft').addEventListener('click', clearDraft);
    document.addEventListener('change', saveDraft);
    document.addEventListener('input', (event) => {{
      if (event.target.tagName === 'TEXTAREA') {{
        saveDraft();
      }}
    }});
    loadDraft();
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", help="Evaluation result jsonl file")
    parser.add_argument("--model", default="gpt-5.4", help="Seed model to prefill")
    parser.add_argument("--seed-json", help="Optional existing goldset JSON to prefill from")
    parser.add_argument("--out", help="Output HTML path")
    args = parser.parse_args()

    input_path = Path(args.jsonl).expanduser().resolve()
    seed_json_path = (
        Path(args.seed_json).expanduser().resolve()
        if args.seed_json is not None
        else None
    )
    output_path = (
        Path(args.out).expanduser().resolve()
        if args.out is not None
        else _default_output_path(input_path, args.model)
    )

    records = [
        record
        for record in _load_records(input_path)
        if record.model == args.model and record.status.ok and record.run is not None
    ]
    if not records:
        raise ValueError(f"No successful records found for model: {args.model}")

    seed_items = _load_seed_items(seed_json_path)
    output_path.write_text(
        _render_html(records, args.model, input_path, seed_json_path, seed_items)
    )
    print(output_path)


if __name__ == "__main__":
    main()
