from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from html import escape
from pathlib import Path
from statistics import median
from typing import Any


JsonDict = dict[str, Any]


def _bool_label(value: object) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    return "null"


def _variant_key(record: JsonDict) -> tuple[str, object]:
    return (str(record["model"]), record.get("thinking"))


def _variant_label(model: str, thinking: object) -> str:
    return f"{model} [thinking={_bool_label(thinking)}]"


def _load_rows(path: Path) -> list[JsonDict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _safe_float(value: object) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return float(str(value))


def _safe_int(value: object) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    return int(str(value))


def _pct(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "-"
    return f"{numerator / denominator * 100:.1f}%"


def _sec(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}s"


def _usd(value: float) -> str:
    if value == 0.0:
        return "$0.0000"
    if value < 0.01:
        return f"${value:.4f}"
    return f"${value:.3f}"


def _summarize(rows: list[JsonDict]) -> list[JsonDict]:
    groups: dict[tuple[str, object], list[JsonDict]] = defaultdict(list)
    for row in rows:
        groups[_variant_key(row)].append(row)

    summaries: list[JsonDict] = []
    for (model, thinking), records in sorted(groups.items()):
        ok_records = [record for record in records if ((record.get("status") or {}).get("ok") is True)]
        failed_records = [record for record in records if ((record.get("status") or {}).get("ok") is not True)]
        latencies = [_safe_float(record.get("latency_sec")) for record in ok_records]
        input_tokens = [
            _safe_int((((record.get("run") or {}).get("usage") or {}).get("input_tokens")))
            for record in ok_records
        ]
        output_tokens = [
            _safe_int((((record.get("run") or {}).get("usage") or {}).get("output_tokens")))
            for record in ok_records
        ]
        costs = [
            _safe_float((((record.get("run") or {}).get("price") or {}).get("total_cost_usd")))
            for record in ok_records
        ]
        detail_levels = Counter(
            str((((record.get("run") or {}).get("detail_level"))))
            for record in ok_records
            if (record.get("run") or {}).get("detail_level") is not None
        )
        error_messages = Counter(
            str(((record.get("status") or {}).get("error")))
            for record in failed_records
            if ((record.get("status") or {}).get("error")) is not None
        )

        summaries.append(
            {
                "model": model,
                "thinking": thinking,
                "label": _variant_label(model, thinking),
                "total": len(records),
                "ok": len(ok_records),
                "failed": len(failed_records),
                "ok_rate": (len(ok_records) / len(records)) if records else 0.0,
                "avg_latency_sec": (sum(latencies) / len(latencies)) if latencies else None,
                "median_latency_sec": median(latencies) if latencies else None,
                "max_latency_sec": max(latencies) if latencies else None,
                "total_input_tokens": sum(input_tokens),
                "total_output_tokens": sum(output_tokens),
                "avg_input_tokens": (sum(input_tokens) / len(input_tokens)) if input_tokens else None,
                "avg_output_tokens": (sum(output_tokens) / len(output_tokens)) if output_tokens else None,
                "total_cost_usd": sum(costs),
                "avg_cost_usd": (sum(costs) / len(costs)) if costs else 0.0,
                "detail_levels": detail_levels,
                "error_messages": error_messages,
            }
        )
    return summaries


def _table_rows(summaries: list[JsonDict]) -> str:
    parts: list[str] = []
    for item in summaries:
        detail_text = ", ".join(
            f"{level}:{count}" for level, count in item["detail_levels"].most_common()
        ) or "-"
        error_text = "<br>".join(
            escape(message) + f" ({count})"
            for message, count in item["error_messages"].most_common(5)
        ) or "-"
        parts.append(
            "<tr>"
            f"<td>{escape(str(item['label']))}</td>"
            f"<td>{item['total']}</td>"
            f"<td>{item['ok']}</td>"
            f"<td>{item['failed']}</td>"
            f"<td>{_pct(int(item['ok']), int(item['total']))}</td>"
            f"<td>{_sec(item['avg_latency_sec'])}</td>"
            f"<td>{_sec(item['median_latency_sec'])}</td>"
            f"<td>{_sec(item['max_latency_sec'])}</td>"
            f"<td>{_usd(float(item['avg_cost_usd']))}</td>"
            f"<td>{_usd(float(item['total_cost_usd']))}</td>"
            f"<td>{int(item['total_input_tokens'])}</td>"
            f"<td>{int(item['total_output_tokens'])}</td>"
            f"<td>{escape(detail_text)}</td>"
            f"<td>{error_text}</td>"
            "</tr>"
        )
    return "\n".join(parts)


def _render_html(input_path: Path, summaries: list[JsonDict], total_rows: int) -> str:
    best_success = max(summaries, key=lambda item: (float(item["ok_rate"]), -int(item["failed"])), default=None)
    fastest = min(
        [item for item in summaries if item["avg_latency_sec"] is not None],
        key=lambda item: float(item["avg_latency_sec"]),
        default=None,
    )
    cheapest = min(summaries, key=lambda item: float(item["avg_cost_usd"]), default=None)

    highlights = [
        f"対象レコード数: {total_rows}",
        f"比較モデル数: {len(summaries)}",
    ]
    if best_success is not None:
        highlights.append(
            f"成功率最高: {best_success['label']} ({_pct(int(best_success['ok']), int(best_success['total']))})"
        )
    if fastest is not None:
        highlights.append(
            f"平均レイテンシ最速: {fastest['label']} ({_sec(fastest['avg_latency_sec'])})"
        )
    if cheapest is not None:
        highlights.append(
            f"平均コスト最安: {cheapest['label']} ({_usd(float(cheapest['avg_cost_usd']))})"
        )

    return f"""<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>image understanding analysis</title>
  <style>
    body {{
      font-family: Helvetica, Arial, sans-serif;
      line-height: 1.45;
      margin: 24px auto 48px;
      max-width: 1400px;
      padding: 0 20px;
      color: #222;
    }}
    h1, h2 {{
      line-height: 1.2;
    }}
    .meta {{
      color: #555;
      margin-bottom: 18px;
    }}
    .highlights {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 10px;
      margin: 18px 0 24px;
    }}
    .card {{
      border: 1px solid #ddd;
      border-radius: 8px;
      padding: 12px 14px;
      background: #fafafa;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
    }}
    th, td {{
      border: 1px solid #ddd;
      padding: 8px 10px;
      vertical-align: top;
      font-size: 13px;
      overflow-wrap: anywhere;
    }}
    th {{
      background: #f3f4f6;
      text-align: left;
      position: sticky;
      top: 0;
    }}
    .table-wrap {{
      overflow: auto;
      border: 1px solid #ddd;
      border-radius: 8px;
    }}
  </style>
</head>
<body>
  <h1>image understanding analysis</h1>
  <div class="meta">元ファイル: {escape(str(input_path))}</div>
  <div class="highlights">
    {"".join(f'<div class="card">{escape(text)}</div>' for text in highlights)}
  </div>
  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th style="width: 220px;">model</th>
          <th>total</th>
          <th>ok</th>
          <th>failed</th>
          <th>ok rate</th>
          <th>avg latency</th>
          <th>median latency</th>
          <th>max latency</th>
          <th>avg cost</th>
          <th>total cost</th>
          <th>input tokens</th>
          <th>output tokens</th>
          <th style="width: 150px;">detail levels</th>
          <th style="width: 320px;">top errors</th>
        </tr>
      </thead>
      <tbody>
        {_table_rows(summaries)}
      </tbody>
    </table>
  </div>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", help="Image understanding result jsonl file")
    parser.add_argument("--out", help="Output HTML path")
    args = parser.parse_args()

    input_path = Path(args.jsonl).expanduser().resolve()
    output_path = (
        Path(args.out).expanduser().resolve()
        if args.out is not None
        else input_path.with_name(f"{input_path.stem}.analysis.html")
    )

    rows = _load_rows(input_path)
    summaries = _summarize(rows)
    output_path.write_text(_render_html(input_path, summaries, len(rows)), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
