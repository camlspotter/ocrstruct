from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from html import escape
from pathlib import Path
from statistics import median
from typing import Any


STAFF_PAGE2_IMAGE_COUNT = 3985
JsonDict = dict[str, Any]


def item_key_from_ref(ref: JsonDict) -> str:
    middle_json_path = str(ref["middle_json_path"])
    page_idx = int(ref["page_idx"])
    block_index = ref.get("block_index")
    image_path = str(ref["image_path"])
    return f"{middle_json_path}::{page_idx}::{block_index}::{image_path}"


def rag_ord(value: str) -> int:
    return {"none": 0, "low": 1, "medium": 2, "high": 3}[value]


def detail_ord(value: str) -> int:
    return {"skip": 0, "short": 1, "long": 2, "extract_text": 3}[value]


def distance_score(pred: int, gold: int) -> float:
    return 1 - abs(pred - gold) / 3


def thinking_rank(value: object) -> int:
    if value is None:
        return 0
    if value is False:
        return 1
    return 2


def variant_key(record: JsonDict) -> tuple[str, object]:
    return (str(record["model"]), record.get("thinking"))


def variant_label(model: str, thinking: object) -> str:
    if thinking is None:
        return model
    return f"{model} [thinking={'true' if thinking else 'false'}]"


def short_label(model: str, thinking: object) -> str:
    base = model.split("/")[-1] if model.startswith("Qwen/") else model
    if thinking is None:
        return base
    return f"{base} [thinking={'true' if thinking else 'false'}]"


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def sec(value: float) -> str:
    return f"{value:.2f}s"


def usd(value: float) -> str:
    return f"${value:.4f}" if value < 0.01 else f"${value:.3f}"


def usd2(value: float) -> str:
    return f"${value:.2f}"


def hours_from_seconds(value: float) -> str:
    return f"{value / 3600:.1f}時間"


def confusion_text(counter: Counter[tuple[str, str]]) -> str:
    if not counter:
        return "-"
    return "<br>".join(
        f"`{gold}`→`{pred}` ({count})"
        for (gold, pred), count in counter.most_common(5)
    )


def render_scatter_svg(path: Path, title: str, points: list[JsonDict]) -> None:
    width = 1000
    height = 560
    ml, mr, mt, mb = 90, 40, 50, 120
    plot_w = width - ml - mr
    plot_h = height - mt - mb
    xs = [float(point["x"]) for point in points]
    ys = [float(point["y"]) for point in points]
    xmin, xmax = 0.0, max(xs) * 1.08
    ymin, ymax = 0.0, max(ys) * 1.15 if max(ys) > 0 else 1.0
    yfmt = (lambda value: f"{value:.4f}") if ymax < 0.01 else (lambda value: f"{value:.2f}")

    def sx(x: float) -> float:
        return ml + (x - xmin) / (xmax - xmin or 1) * plot_w

    def sy(y: float) -> float:
        return mt + plot_h - (y - ymin) / (ymax - ymin or 1) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:-apple-system,BlinkMacSystemFont,"Hiragino Sans",sans-serif;fill:#222} .small{font-size:11px} .axis{stroke:#666;stroke-width:1} .grid{stroke:#ddd;stroke-width:1} .pt{fill:#2f6db3}</style>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-size="20">{escape(title)}</text>',
    ]
    for index in range(6):
        gy = mt + plot_h * index / 5
        gv = ymax * (5 - index) / 5
        parts.append(
            f'<line class="grid" x1="{ml}" y1="{gy:.1f}" x2="{width - mr}" y2="{gy:.1f}"/>'
        )
        parts.append(
            f'<text class="small" x="{ml - 8}" y="{gy + 4:.1f}" text-anchor="end">{yfmt(gv)}</text>'
        )
    for index in range(6):
        gx = ml + plot_w * index / 5
        gv = xmax * index / 5
        parts.append(
            f'<line class="grid" x1="{gx:.1f}" y1="{mt}" x2="{gx:.1f}" y2="{mt + plot_h}"/>'
        )
        parts.append(
            f'<text class="small" x="{gx:.1f}" y="{mt + plot_h + 18}" text-anchor="middle">{gv:.1f}</text>'
        )
    parts.append(
        f'<line class="axis" x1="{ml}" y1="{mt + plot_h}" x2="{width - mr}" y2="{mt + plot_h}"/>'
    )
    parts.append(
        f'<line class="axis" x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt + plot_h}"/>'
    )
    parts.append(f'<text x="{width / 2}" y="{height - 24}" text-anchor="middle">平均レイテンシ (秒/画像)</text>')
    parts.append(
        f'<text x="24" y="{height / 2}" text-anchor="middle" transform="rotate(-90 24 {height / 2})">平均コスト (USD/画像)</text>'
    )
    for point in points:
        x = sx(float(point["x"]))
        y = sy(float(point["y"]))
        label = escape(str(point["label"]))
        parts.append(f'<circle class="pt" cx="{x:.1f}" cy="{y:.1f}" r="5"/>')
        parts.append(f'<text class="small" x="{x + 7:.1f}" y="{y - 7:.1f}">{label}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts))


def render_barh_svg(
    path: Path,
    title: str,
    rows: list[JsonDict],
    value_key: str,
    label_key: str,
    xmax: float = 1.0,
) -> None:
    width = 1100
    row_h = 34
    height = 80 + row_h * len(rows) + 30
    ml, mr, mt, mb = 300, 40, 50, 30
    plot_w = width - ml - mr
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:-apple-system,BlinkMacSystemFont,"Hiragino Sans",sans-serif;fill:#222} .small{font-size:12px} .bar{fill:#4c78a8} .axis{stroke:#666;stroke-width:1} .grid{stroke:#ddd;stroke-width:1}</style>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-size="20">{escape(title)}</text>',
    ]
    for index in range(6):
        x = ml + plot_w * index / 5
        value = xmax * index / 5
        parts.append(
            f'<line class="grid" x1="{x:.1f}" y1="{mt}" x2="{x:.1f}" y2="{height - mb}"/>'
        )
        parts.append(
            f'<text class="small" x="{x:.1f}" y="{height - 8}" text-anchor="middle">{value:.1f}</text>'
        )
    for index, row in enumerate(rows):
        y = mt + index * row_h + 8
        value = float(row[value_key])
        width_px = plot_w * value / xmax
        parts.append(
            f'<text class="small" x="{ml - 10}" y="{y + 12:.1f}" text-anchor="end">{escape(str(row[label_key]))}</text>'
        )
        parts.append(
            f'<rect class="bar" x="{ml}" y="{y:.1f}" width="{width_px:.1f}" height="20" rx="3"/>'
        )
        parts.append(
            f'<text class="small" x="{ml + width_px + 6:.1f}" y="{y + 14:.1f}">{value:.3f}</text>'
        )
    parts.append(
        f'<line class="axis" x1="{ml}" y1="{height - mb}" x2="{width - mr}" y2="{height - mb}"/>'
    )
    parts.append("</svg>")
    path.write_text("\n".join(parts))


def render_grouped_bar_svg(path: Path, title: str, rows: list[JsonDict]) -> None:
    width = 1300
    height = 620
    ml, mr, mt, mb = 70, 40, 50, 220
    plot_w = width - ml - mr
    plot_h = height - mt - mb
    colors = ["#4c78a8", "#f58518", "#54a24b"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:-apple-system,BlinkMacSystemFont,"Hiragino Sans",sans-serif;fill:#222} .small{font-size:11px} .axis{stroke:#666;stroke-width:1} .grid{stroke:#ddd;stroke-width:1}</style>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-size="20">{escape(title)}</text>',
    ]
    for index in range(6):
        gy = mt + plot_h * index / 5
        value = (5 - index) / 5
        parts.append(
            f'<line class="grid" x1="{ml}" y1="{gy:.1f}" x2="{width - mr}" y2="{gy:.1f}"/>'
        )
        parts.append(
            f'<text class="small" x="{ml - 8}" y="{gy + 4:.1f}" text-anchor="end">{value:.1f}</text>'
        )
    group_w = plot_w / max(len(rows), 1)
    bar_w = group_w / 4
    for index, row in enumerate(rows):
        gx = ml + index * group_w + group_w / 2
        values = [
            float(row["kind_accuracy"]),
            float(row["rag_distance_score"]),
            float(row["detail_distance_score"]),
        ]
        for offset, value in enumerate(values):
            h = plot_h * value
            x = gx + (offset - 1) * bar_w - bar_w / 2
            y = mt + plot_h - h
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{colors[offset]}" rx="2"/>'
            )
        label = escape(str(row["short_label"]))
        parts.append(
            f'<text class="small" x="{gx:.1f}" y="{height - 100}" text-anchor="end" transform="rotate(-35 {gx:.1f} {height - 100})">{label}</text>'
        )
    parts.append(
        f'<line class="axis" x1="{ml}" y1="{mt + plot_h}" x2="{width - mr}" y2="{mt + plot_h}"/>'
    )
    legend_x = width - 320
    legend_y = 60
    legends = [
        ("kind 一致率", colors[0]),
        ("rag 距離スコア", colors[1]),
        ("detail 距離スコア", colors[2]),
    ]
    for index, (label, color) in enumerate(legends):
        y = legend_y + index * 24
        parts.append(f'<rect x="{legend_x}" y="{y}" width="14" height="14" fill="{color}"/>')
        parts.append(f'<text class="small" x="{legend_x + 22}" y="{y + 12}">{escape(label)}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts))


def load_rows(path: Path) -> list[JsonDict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_metrics(
    rows: list[JsonDict],
    gold_by_key: dict[str, JsonDict],
) -> tuple[list[JsonDict], list[JsonDict], set[str], list[JsonDict]]:
    ok_rows = [row for row in rows if row["status"]["ok"] and row.get("run") is not None]  # type: ignore[index]
    variants = sorted({variant_key(row) for row in ok_rows}, key=lambda x: (x[0], thinking_rank(x[1])))
    item_counts: dict[str, set[tuple[str, object]]] = defaultdict(set)
    for row in ok_rows:
        item_counts[item_key_from_ref(row["ref"])] .add(variant_key(row))  # type: ignore[index]
    complete_keys = {key for key, seen in item_counts.items() if seen == set(variants)}
    filtered = [row for row in ok_rows if item_key_from_ref(row["ref"]) in complete_keys]  # type: ignore[index]
    by_variant: dict[tuple[str, object], list[JsonDict]] = defaultdict(list)
    for row in filtered:
        by_variant[variant_key(row)].append(row)

    metrics: list[JsonDict] = []
    for variant in variants:
        records = by_variant[variant]
        latencies = [float(record["latency_sec"]) for record in records]
        costs = [
            float((((record.get("run") or {}).get("price") or {}).get("total_cost_usd") or 0.0))
            for record in records
        ]
        kind_hits: list[float] = []
        rag_dists: list[int] = []
        detail_dists: list[int] = []
        rag_scores: list[float] = []
        detail_scores: list[float] = []
        all_exact_hits: list[float] = []
        kind_confusions: Counter[tuple[str, str]] = Counter()
        for record in records:
            gold = gold_by_key[item_key_from_ref(record["ref"])]  # type: ignore[index]
            pred = record["run"]  # type: ignore[index]
            pred_kind = str(pred["kind"])
            gold_kind = str(gold["kind"])
            pred_rag = str(pred["rag_value"])
            gold_rag = str(gold["rag_value"])
            pred_detail = str(pred["detail_level"])
            gold_detail = str(gold["detail_level"])
            kind_hit = pred_kind == gold_kind
            rag_dist = abs(rag_ord(pred_rag) - rag_ord(gold_rag))
            detail_dist = abs(detail_ord(pred_detail) - detail_ord(gold_detail))
            kind_hits.append(1.0 if kind_hit else 0.0)
            rag_dists.append(rag_dist)
            detail_dists.append(detail_dist)
            rag_scores.append(distance_score(rag_ord(pred_rag), rag_ord(gold_rag)))
            detail_scores.append(distance_score(detail_ord(pred_detail), detail_ord(gold_detail)))
            all_exact_hits.append(
                1.0 if kind_hit and pred_rag == gold_rag and pred_detail == gold_detail else 0.0
            )
            if not kind_hit:
                kind_confusions[(gold_kind, pred_kind)] += 1
        kind_score = sum(kind_hits) / len(records)
        rag_score = sum(rag_scores) / len(records)
        detail_score = sum(detail_scores) / len(records)
        model, thinking = variant
        metrics.append(
            {
                "variant": variant,
                "label": variant_label(model, thinking),
                "short_label": short_label(model, thinking),
                "avg_latency_sec": sum(latencies) / len(latencies),
                "median_latency_sec": median(latencies),
                "avg_cost_usd": sum(costs) / len(costs),
                "total_cost_usd": sum(costs),
                "kind_accuracy": kind_score,
                "rag_mean_distance": sum(rag_dists) / len(records),
                "rag_distance_score": rag_score,
                "detail_mean_distance": sum(detail_dists) / len(records),
                "detail_distance_score": detail_score,
                "overall_score": 0.4 * kind_score + 0.3 * rag_score + 0.3 * detail_score,
                "all_exact_accuracy": sum(all_exact_hits) / len(records),
                "kind_confusions": kind_confusions,
            }
        )
    metrics_by_score = sorted(metrics, key=lambda item: float(item["overall_score"]), reverse=True)
    metrics_by_variant = sorted(
        metrics, key=lambda item: (tuple(item["variant"])[0], thinking_rank(tuple(item["variant"])[1]))
    )
    return metrics_by_score, metrics_by_variant, complete_keys, filtered


def build_markdown(
    metrics_by_score: list[JsonDict],
    metrics_by_variant: list[JsonDict],
    complete_keys: set[str],
    filtered: list[dict[str, object]],
    results_name: str,
    gold_name: str,
) -> str:
    best_overall = max(metrics_by_score, key=lambda item: float(item["overall_score"]))
    fastest = min(metrics_by_score, key=lambda item: float(item["avg_latency_sec"]))
    qwen27_default = next(
        item for item in metrics_by_variant if tuple(item["variant"]) == ("Qwen/Qwen3.6-27B-FP8", None)
    )
    qwen27_off = next(
        item for item in metrics_by_variant if tuple(item["variant"]) == ("Qwen/Qwen3.6-27B-FP8", False)
    )
    qwen35_off = next(
        item for item in metrics_by_variant if tuple(item["variant"]) == ("Qwen/Qwen3.6-35B-A3B-FP8", False)
    )

    lines = [
        "# VLM による画像分類実験",
        "",
        f"計測データ: `{results_name}`",
        f"正解データ: `{gold_name}`",
        "",
        "## Abstract",
        "",
        "Multimodal RAG を作成するにあたり、文書から画像を取り出し、VLM によりそれらの",
        "説明テキストを抽出したい。その前段階の screening として、画像の説明テキストを",
        "コストと時間をかけて得る「テキスト化価値」があるかどうかの判別を",
        "高速、低コストで行うこととした。",
        "",
        "本実験では、ROIS にある文書中の約 100個の画像に対し、複数のモデルを使用して",
        "テキスト化価値の推定を行い、モデルの評価を行った。",
        "",
        "## テキスト化価値",
        "",
        "テキスト化価値とは、画像について追加の説明文や OCR 結果を生成し、",
        "RAG に登録するだけの価値があるかどうかを指す。",
        "",
        "`kind`: 画像の種類。たとえば `diagram`, `text_as_image`, `seal`, `code_symbol` など。",
        "",
        "`rag_value` の値は次のように解釈する:",
        "",
        "- `high`: 画像そのものが情報の担い手であり、説明文や抽出テキストを RAG に入れる価値が高い。",
        "- `medium`: 一定の情報価値はあるが、文書本文だけで大筋が把握できる可能性もある。",
        "- `low`: 補助的な価値はあるが、RAG への寄与は限定的である。",
        "- `none`: 説明文や OCR を追加しても、RAG 上の価値はほとんど期待できない。",
        "",
        "`detail_level` の値は、screening 後にどこまでコストをかけて調べるべきかを表す:",
        "",
        "- `skip`: 追加調査を行わない。",
        "- `short`: 短い説明文だけを付ければ十分である。",
        "- `long`: 構造や意味関係を含めた、やや詳しい説明文を作る価値がある。",
        "- `extract_text`: 詳しい説明に加え、画像内テキストそのものを OCR などで抽出する価値が高い。",
        "",
        "この2つは似ているが、役割が異なる。`rag_value` は「そもそも価値があるか」を表し、",
        "`detail_level` は「価値があるとして、どこまで調べるか」を表す。",
        "",
        "## 対象概要",
        "",
        f"画像: ROIS本部に存在する文書から人間が選んだ画像 {len(complete_keys)} 枚",
        "",
        "モデル",
        "",
    ]
    for item in metrics_by_variant:
        lines.append(f"- {item['label']}")
    lines += [
        "",
        "ここで `thinking=null` は、thinking を明示指定せず、",
        "モデル側のデフォルト動作に任せた条件を意味する。",
        "Qwen 3.6 系では、この条件は実質的に `thinking=true` と解釈して差し支えない。",
        "Qwen モデルについては DGX Spark 互換機を利用し、そこでの処理時間を計測し、費用は0とした。",
        "",
        "OpenAI モデルについては `reasoning_effort` の指定を行っていない。",
        "一部モデルでは既定値が `none` と明記されているが、全モデルで未指定時の挙動を断定はしない。",
        "",
        f"これらを組み合わせ、合計 {len(filtered)} の結果を得た。",
        "",
        f"計測データ: `{results_name}`",
        "",
        "## 正解データとスコアの定義",
        "",
        "正解データとして、 `gpt-5.4` で得られた回答を人間がチェックし、修正を加えた物を使用する。",
        "",
        f"正解データ: `{gold_name}`",
        "",
        "各モデルの出力を正解データと比べて採点した。",
        "",
        "- `kind` は離散的なラベルなので、正解と完全に一致したかどうかで評価する。",
        "- `rag_value` と `detail_level` は順序付きの尺度なので、以下の距離スコアを使う:",
        "  - 完全一致なら 1.0",
        "  - 1段階ずれなら 0.67",
        "  - 2段階ずれなら 0.33",
        "  - 最大ずれなら 0.0",
        "  式で書くと $1 - |\\mathit{pred} - \\mathit{gold}| / 3$ である。",
        "",
        "最終的な総合スコアは、次の重み付き平均で計算した。",
        "",
        "- `kind` 完全一致: 40%",
        "- `rag_value` 距離スコア: 30%",
        "- `detail_level` 距離スコア: 30%",
        "",
        "正解データの作成経緯から `gpt-5.4` でのスコアが高くなるバイアスが存在している可能性がある。",
        "",
        "## 評価",
        "",
        "### 総合スコア",
        "",
        "![Overall Score](image_eval_gold_overall_score.svg)",
        "",
        "| モデル条件 | 総合スコア | kind 一致率 | rag 距離スコア | detail 距離スコア | 3項目完全一致率 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in metrics_by_score:
        lines.append(
            f"| `{item['label']}` | {float(item['overall_score']):.3f} | {pct(float(item['kind_accuracy']))} | {float(item['rag_distance_score']):.3f} | {float(item['detail_distance_score']):.3f} | {pct(float(item['all_exact_accuracy']))} |"
        )
    lines += [
        "",
        "### 項目別スコア",
        "",
        "![Gold Axis Accuracy](image_eval_gold_axis_accuracy.svg)",
        "",
        "| モデル条件 | kind 一致率 | rag 平均ずれ | rag 距離スコア | detail 平均ずれ | detail 距離スコア |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in metrics_by_score:
        lines.append(
            f"| `{item['label']}` | {pct(float(item['kind_accuracy']))} | {float(item['rag_mean_distance']):.3f} | {float(item['rag_distance_score']):.3f} | {float(item['detail_mean_distance']):.3f} | {float(item['detail_distance_score']):.3f} |"
        )
    lines += [
        "",
        "### コストとレイテンシ",
        "",
        "![Latency vs Cost](image_eval_latency_cost.svg)",
        "",
        "| モデル条件 | 平均レイテンシ | 中央レイテンシ | 1画像あたり平均コスト | このセット合計コスト |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in metrics_by_variant:
        lines.append(
            f"| `{item['label']}` | {sec(float(item['avg_latency_sec']))} | {sec(float(item['median_latency_sec']))} | {usd(float(item['avg_cost_usd']))} | {usd(float(item['total_cost_usd']))} |"
        )
    lines += [
        "",
        "### 主な所見",
        "",
        f"総合スコアが最も高かったのは `{best_overall['label']}` の {float(best_overall['overall_score']):.3f}。ただし正解データは `gpt-5.4` の結果を元にしているため、バイアスが入っている可能性がある。",
        "",
        f"最速は `{fastest['label']}` だった。",
        "",
        "OpenAI の古いモデルでは意外と時間がかかる傾向がある。",
        "",
        f"Qwen では `Qwen/Qwen3.6-27B-FP8` の `thinking=null`（実質 `thinking=true`）と `thinking=false` で総合スコアはそれぞれ {float(qwen27_default['overall_score']):.3f}, {float(qwen27_off['overall_score']):.3f} と大差なかったにも関わらず、",
        f"時間差が極端で、平均 {float(qwen27_default['avg_latency_sec']):.1f}s と {float(qwen27_off['avg_latency_sec']):.1f}s であることが目立つ。",
        "本 screening タスクでは `thinking=true` の採用意義は見えにくい。",
        "",
        f"`Qwen/Qwen3.6-35B-A3B-FP8 [thinking=false]` は平均 {float(qwen35_off['avg_latency_sec']):.1f}s だった。",
        "",
        "費用に関しては、本レポートの価格定義では open-weight model が最安である。",
        "",
        "### ROIS 文書中の画像 3985件に外挿した概算",
        "",
        "現在、ROIS のローカル文書中には screening 対象の画像が 3985 件ある。",
        "今回の実験の 1件あたり平均値を単純外挿すると、これらを処理する際の目安は次のとおりである:",
        "",
        "| モデル条件 | 3985件の概算コスト | 3985件を直列実行した場合の概算時間 |",
        "|---|---:|---:|",
    ]
    for item in metrics_by_variant:
        lines.append(
            f"| `{item['label']}` | {usd2(float(item['avg_cost_usd']) * STAFF_PAGE2_IMAGE_COUNT)} | {hours_from_seconds(float(item['avg_latency_sec']) * STAFF_PAGE2_IMAGE_COUNT)} |"
        )
    lines += [
        "",
        "時間については OpenAI モデルについては並列処理を行うことで短縮することができる。",
        "Open weight model については ROIS 本部での機材の関係上、並列化は難しい。",
        "",
        "## 推奨",
        "",
        f"総合評価で見ると、第一候補は `{best_overall['label']}`。",
        f"ローカル GPU で速度と品質のバランスが最も良いのは `{qwen27_off['label']}`。平均 {float(qwen27_off['avg_latency_sec']):.1f}s で、総合スコアは {float(qwen27_off['overall_score']):.3f}。",
        "`Qwen/Qwen3.6-27B-FP8` は `thinking` 条件で速度が大きく変わる。本 screening タスクでは Thinking はほぼ役に立っていない。",
        "",
        "## 付録: kind の主な取り違え",
        "",
        "| モデル条件 | 主な取り違え |",
        "|---|---|",
    ]
    for item in metrics_by_score:
        lines.append(f"| `{item['label']}` | {confusion_text(item['kind_confusions'])} |")
    return "\n".join(lines) + "\n"


def build_html(markdown: str, markdown_name: str) -> str:
    lines = markdown.splitlines()
    body: list[str] = []
    in_ul = False
    in_ol = False
    table_lines: list[str] = []

    def flush_table() -> None:
        nonlocal table_lines
        if not table_lines:
            return
        header = [cell.strip() for cell in table_lines[0].strip("|").split("|")]
        body.append("<table>")
        body.append("<thead><tr>" + "".join(f"<th>{escape(cell)}</th>" for cell in header) + "</tr></thead>")
        body.append("<tbody>")
        for row in table_lines[2:]:
            cols = [cell.strip() for cell in row.strip("|").split("|")]
            body.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cols) + "</tr>")
        body.append("</tbody></table>")
        table_lines = []

    for line in lines:
        if line.startswith("|"):
            table_lines.append(line)
            continue
        flush_table()
        if line.startswith("- "):
            if not in_ul:
                body.append("<ul>")
                in_ul = True
            body.append(f"<li>{escape(line[2:])}</li>")
            continue
        if in_ul:
            body.append("</ul>")
            in_ul = False
        if line.startswith(("1. ", "2. ", "3. ", "4. ", "5. ")):
            if not in_ol:
                body.append("<ol>")
                in_ol = True
            body.append(f"<li>{escape(line[3:])}</li>")
            continue
        if in_ol:
            body.append("</ol>")
            in_ol = False
        if line.startswith("# "):
            body.append(f"<h1>{escape(line[2:])}</h1>")
        elif line.startswith("## "):
            body.append(f"<h2>{escape(line[3:])}</h2>")
        elif line.startswith("### "):
            body.append(f"<h3>{escape(line[4:])}</h3>")
        elif line.startswith("!["):
            src = line.split("](", 1)[1][:-1]
            body.append(f'<p><img src="{escape(src)}" alt="chart"></p>')
        elif line:
            body.append(f"<p>{line}</p>")
        else:
            body.append("")
    flush_table()
    if in_ul:
        body.append("</ul>")
    if in_ol:
        body.append("</ol>")

    return (
        '<!doctype html><html lang="ja"><head><meta charset="utf-8"><title>image eval analysis</title>'
        '<style>body{font-family:-apple-system,BlinkMacSystemFont,"Hiragino Sans",sans-serif;line-height:1.6;'
        'margin:24px auto;max-width:1100px;padding:0 16px}table{border-collapse:collapse;width:100%;margin:16px 0;'
        'font-size:14px}th,td{border:1px solid #ccc;padding:6px 8px;text-align:left;vertical-align:top}'
        'th{background:#f6f6f6}img{max-width:100%;height:auto;border:1px solid #ddd}'
        'code{background:#f4f4f4;padding:1px 4px;border-radius:4px}</style></head><body>'
        + "".join(body)
        + f'<p>表を含む完全版は <a href="{escape(markdown_name)}" target="_blank" rel="noopener noreferrer">{escape(markdown_name)}</a> を参照してください。</p>'
        + "</body></html>"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="screening/image_eval_results_v2.jsonl")
    parser.add_argument("--gold", default="screening/image_eval_results.gpt-5.4.goldset.json")
    parser.add_argument("--out-md", default="screening/image_eval_analysis.md")
    parser.add_argument("--out-html", default="screening/image_eval_analysis.html")
    parser.add_argument("--latency-svg", default="screening/image_eval_latency_cost.svg")
    parser.add_argument("--overall-svg", default="screening/image_eval_gold_overall_score.svg")
    parser.add_argument("--axis-svg", default="screening/image_eval_gold_axis_accuracy.svg")
    args = parser.parse_args()

    results_path = Path(args.results)
    gold_path = Path(args.gold)
    out_md_path = Path(args.out_md)
    out_html_path = Path(args.out_html)
    latency_svg_path = Path(args.latency_svg)
    overall_svg_path = Path(args.overall_svg)
    axis_svg_path = Path(args.axis_svg)

    gold_data: JsonDict = json.loads(gold_path.read_text())
    gold_by_key = {str(item["item_key"]): item for item in gold_data["items"]}
    rows = load_rows(results_path)
    metrics_by_score, metrics_by_variant, complete_keys, filtered = build_metrics(rows, gold_by_key)

    render_scatter_svg(
        latency_svg_path,
        "コストとレイテンシ",
        [{"x": item["avg_latency_sec"], "y": item["avg_cost_usd"], "label": item["short_label"]} for item in metrics_by_variant],
    )
    render_barh_svg(
        overall_svg_path,
        "gold set 総合スコア",
        metrics_by_score,
        "overall_score",
        "short_label",
        1.0,
    )
    render_grouped_bar_svg(axis_svg_path, "gold set 項目別スコア", metrics_by_score)

    markdown = build_markdown(
        metrics_by_score,
        metrics_by_variant,
        complete_keys,
        filtered,
        results_path.name,
        gold_path.name,
    )
    out_md_path.write_text(markdown)
    out_html_path.write_text(build_html(markdown, out_md_path.name))


if __name__ == "__main__":
    main()
