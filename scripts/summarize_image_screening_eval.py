from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from ocrstruct.image_understanding import Model
from scripts.run_image_screening_eval import EvalRecord


class Summary(Model):
    model: str
    total: int
    ok: int
    failed: int
    avg_latency_sec: float | None
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: float


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl")
    args = parser.parse_args()

    rows = Path(args.jsonl).read_text().splitlines()
    groups: dict[str, list[EvalRecord]] = defaultdict(list)
    for row in rows:
        if not row.strip():
            continue
        record = EvalRecord.model_validate_json(row)
        groups[record.model].append(record)

    for model, records in groups.items():
        ok_records = [record for record in records if record.status.ok and record.run is not None]
        total_input_tokens = sum(
            record.run.usage.input_tokens or 0
            for record in ok_records
            if record.run is not None and record.run.usage is not None
        )
        total_output_tokens = sum(
            record.run.usage.output_tokens or 0
            for record in ok_records
            if record.run is not None and record.run.usage is not None
        )
        total_cost_usd = sum(
            record.run.price.total_cost_usd or 0.0
            for record in ok_records
            if record.run is not None and record.run.price is not None
        )
        avg_latency_sec = None
        if ok_records:
            avg_latency_sec = sum(record.latency_sec for record in ok_records) / len(ok_records)
        summary = Summary(
            model=model,
            total=len(records),
            ok=len(ok_records),
            failed=len(records) - len(ok_records),
            avg_latency_sec=avg_latency_sec,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            total_cost_usd=total_cost_usd,
        )
        print(summary.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
