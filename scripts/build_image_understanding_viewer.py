from __future__ import annotations

import argparse
import html
from pathlib import Path

from pydantic import Field

from ocrstruct.image_understanding import ImageRef, Model, PriceEstimate, TokenUsage


class LegacyRunStatus(Model):
    ok: bool
    error: str | None = None


class LegacyScreeningRunView(Model):
    kind: str
    rag_value: str
    detail_level: str
    notes: str | None = None
    raw_text: str
    usage: TokenUsage | None = None
    price: PriceEstimate | None = None


class LegacyScreeningSource(Model):
    model: str
    thinking: bool | None = None
    resolved_thinking: bool | None = None
    base_url: str | None = None
    started_at: str | None = None
    latency_sec: float
    run: LegacyScreeningRunView


class LegacyUnderstandingRunView(Model):
    kind: str
    rag_value: str
    detail_level: str
    keywords: list[str] = Field(default_factory=list)
    notes: str | None = None
    short_description: str | None = None
    long_description: str | None = None
    raw_text: str
    usage: TokenUsage | None = None
    price: PriceEstimate | None = None


class LegacyUnderstandingRecord(Model):
    ref: ImageRef
    screening: LegacyScreeningSource
    model: str
    thinking: bool = False
    resolved_thinking: bool = False
    base_url: str | None = None
    started_at: str | None = None
    latency_sec: float
    status: LegacyRunStatus
    run: LegacyUnderstandingRunView | None = None


def _load_records(path: Path) -> list[LegacyUnderstandingRecord]:
    return [
        LegacyUnderstandingRecord.model_validate_json(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def _image_src(ref: ImageRef) -> str:
    image_path = Path(ref.image_path)
    if image_path.is_absolute():
        return str(image_path)
    return str(Path(ref.middle_json_path).parent / "images" / image_path)


def _pdf_href(ref: ImageRef) -> str:
    return f"{ref.pdf_path}#page={ref.page_idx + 1}"


def _text_block(value: str | None) -> str:
    if value is None or value == "":
        return '<span class="empty">null</span>'
    return html.escape(value).replace("\n", "<br>")


def _detail_badge(detail_level: str) -> str:
    class_name = f"badge detail-{detail_level}"
    return f'<span class="{html.escape(class_name)}">{html.escape(detail_level)}</span>'


def _render_item(index: int, record: LegacyUnderstandingRecord) -> str:
    ref = record.ref
    status_text = "ok" if record.status.ok else f"error: {record.status.error}"
    screening = record.screening
    run = record.run
    if run is None:
        body = f'<div class="error-box">{html.escape(status_text)}</div>'
    else:
        body = f"""
    <div class="result-grid">
      <div class="field">
        <div class="field-label">screening</div>
        <div class="pill-row">
          <span class="badge"><strong>kind:</strong>&nbsp;{html.escape(screening.run.kind)}</span>
          <span class="badge">{html.escape(screening.run.rag_value)}</span>
          {_detail_badge(screening.run.detail_level)}
        </div>
        <div class="field-text">{_text_block(screening.run.notes)}</div>
      </div>
      <div class="field compact">
        <div class="field-label">understanding labels</div>
        <div class="pill-row">
          <span class="badge"><strong>kind:</strong>&nbsp;{html.escape(run.kind)}</span>
          <span class="badge"><strong>rag:</strong>&nbsp;{html.escape(run.rag_value)}</span>
          {_detail_badge(run.detail_level)}
        </div>
      </div>
      <div class="field">
        <div class="field-label">understanding keywords</div>
        <div class="pill-row">
          {"".join(f'<span class="badge">{html.escape(keyword)}</span>' for keyword in run.keywords) if run.keywords else '<span class="empty">[]</span>'}
        </div>
      </div>
      <div class="field">
        <div class="field-label">understanding short_description</div>
        <div class="field-text">{_text_block(run.short_description)}</div>
      </div>
      <div class="field">
        <div class="field-label">understanding long_description</div>
        <div class="field-text">{_text_block(run.long_description)}</div>
      </div>
      <div class="field">
        <div class="field-label">understanding notes</div>
        <div class="field-text">{_text_block(run.notes)}</div>
      </div>
      <div class="field compact">
        <div class="field-label">cost / latency</div>
        <div class="field-text">
          <div><strong>latency_sec:</strong> {record.latency_sec:.3f}</div>
          <div><strong>price.total_cost_usd:</strong> {run.price.total_cost_usd if run.price is not None else "null"}</div>
          <div><strong>input_tokens:</strong> {run.usage.input_tokens if run.usage is not None else "null"}</div>
          <div><strong>output_tokens:</strong> {run.usage.output_tokens if run.usage is not None else "null"}</div>
        </div>
      </div>
    </div>
    <details>
      <summary>raw response</summary>
      <pre>{html.escape(run.raw_text)}</pre>
    </details>
"""
    return f"""
  <section class="item">
    <h2>{index + 1}. {html.escape(ref.image_path)}</h2>
    <div class="meta">
      <div><strong>status:</strong> {html.escape(status_text)}</div>
      <div><strong>understanding model:</strong> {html.escape(record.model)}</div>
      <div><strong>understanding thinking:</strong> {html.escape(str(record.thinking).lower())}</div>
      <div><strong>screening model:</strong> {html.escape(screening.model)}</div>
      <div><strong>screening thinking:</strong> {html.escape(str(screening.thinking).lower())}</div>
      <div><strong>pdf_path:</strong> <a href="{html.escape(_pdf_href(ref), quote=True)}" target="_blank" rel="noopener noreferrer">{html.escape(ref.pdf_path)}</a></div>
      <div><strong>middle_json_path:</strong> <a href="{html.escape(ref.middle_json_path, quote=True)}" target="_blank" rel="noopener noreferrer">{html.escape(ref.middle_json_path)}</a></div>
      <div><strong>page_idx:</strong> {ref.page_idx}</div>
      <div><strong>block_index:</strong> {ref.block_index}</div>
      <div><strong>block_type:</strong> {html.escape(ref.block_type)}</div>
    </div>
    <img src="{html.escape(_image_src(ref), quote=True)}" alt="{html.escape(ref.image_path)}">
    <div class="context-stack">
      <div class="context-box"><strong>caption</strong><div>{_text_block(ref.caption)}</div></div>
      <div class="context-box"><strong>section_title</strong><div>{_text_block(ref.section_title)}</div></div>
      <div class="context-box"><strong>nearby_text_before</strong><div>{_text_block(ref.nearby_text_before)}</div></div>
      <div class="context-box"><strong>nearby_text_after</strong><div>{_text_block(ref.nearby_text_after)}</div></div>
    </div>
{body}
  </section>
"""


def _default_output_path(input_path: Path) -> Path:
    return input_path.with_suffix(".html")


def _render_html(records: list[LegacyUnderstandingRecord], input_path: Path) -> str:
    items = "".join(_render_item(index, record) for index, record in enumerate(records))
    models = sorted({record.model for record in records})
    return f"""<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>image understanding viewer</title>
  <style>
    :root {{
      --border: #d9d9d9;
      --muted: #666;
      --bg: #fafafa;
      --accent: #0f766e;
      --skip: #e5e7eb;
      --short: #dbeafe;
      --long: #fef3c7;
      --extract: #dcfce7;
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
    .context-stack,
    .result-grid {{
      display: grid;
      gap: 8px;
      margin-top: 8px;
      margin-bottom: 10px;
    }}
    .context-box,
    .field,
    .error-box {{
      border: 1px solid var(--border);
      border-radius: 6px;
      background: white;
      padding: 8px 10px;
      font-size: 13px;
    }}
    .field-label,
    .context-box strong {{
      display: block;
      font-weight: 700;
      margin-bottom: 4px;
    }}
    .context-stack {{
      gap: 4px;
      margin-top: 6px;
      margin-bottom: 8px;
    }}
    .context-box {{
      padding: 5px 7px;
      font-size: 11px;
      color: var(--muted);
      line-height: 1.35;
    }}
    .context-box strong {{
      margin-bottom: 2px;
      font-size: 10px;
      letter-spacing: 0.02em;
      text-transform: lowercase;
    }}
    .context-box div {{
      max-height: calc(1.35em * 3);
      overflow-y: auto;
      padding-right: 2px;
    }}
    .field-text {{
      white-space: normal;
      overflow-wrap: anywhere;
    }}
    .mono {{
      font-family: Menlo, Monaco, monospace;
      white-space: pre-wrap;
    }}
    .pill-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-bottom: 6px;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      border: 1px solid var(--border);
      border-radius: 999px;
      padding: 4px 8px;
      background: #fff;
      font-size: 13px;
    }}
    .detail-skip {{ background: var(--skip); }}
    .detail-short {{ background: var(--short); }}
    .detail-long {{ background: var(--long); }}
    .detail-extract_text {{ background: var(--extract); }}
    .empty {{
      color: var(--muted);
      font-style: italic;
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
      <strong>image understanding viewer</strong>
      <span>items: {len(records)}</span>
      <span>understanding models: {html.escape(", ".join(models))}</span>
    </div>
  </div>

  <h1>image understanding viewer</h1>
  <p>元ファイル: {html.escape(str(input_path))}</p>

{items}
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", help="Image understanding result jsonl file")
    parser.add_argument("--model", help="Optional understanding model filter")
    parser.add_argument("--out", help="Output HTML path")
    args = parser.parse_args()

    input_path = Path(args.jsonl).expanduser().resolve()
    output_path = (
        Path(args.out).expanduser().resolve()
        if args.out is not None
        else _default_output_path(input_path)
    )

    records = _load_records(input_path)
    if args.model is not None:
        records = [record for record in records if record.model == args.model]
    if not records:
        raise ValueError("No records found for the requested filter.")

    output_path.write_text(_render_html(records, input_path))
    print(output_path)


if __name__ == "__main__":
    main()
