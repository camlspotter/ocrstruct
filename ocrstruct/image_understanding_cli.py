from __future__ import annotations

import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv

from ocrstruct.image_understanding import (
    iter_understanding_records_from_screening,
    load_completed_understanding_keys,
    load_pricing_overrides,
    load_screening_records_jsonl,
    pricing_for_model,
    understanding_record_key,
)


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(
        prog="ocrstruct-understanding",
        description="Generate image understanding JSONL from image screening JSONL.",
    )
    parser.add_argument("--screening-results", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument(
        "--screening-thinking",
        choices=["true", "false"],
        help="Filter source screening results by thinking flag.",
    )
    parser.add_argument("--base-url")
    parser.add_argument("--api-key")
    parser.add_argument("--pricing-json")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip already completed (understanding model, screening source, image) rows.",
    )
    thinking_group = parser.add_mutually_exclusive_group()
    thinking_group.add_argument(
        "--thinking",
        dest="thinking",
        action="store_const",
        const=True,
        help="Force thinking/reasoning on.",
    )
    thinking_group.add_argument(
        "--no-thinking",
        dest="thinking",
        action="store_const",
        const=False,
        help="Force thinking/reasoning off.",
    )
    parser.set_defaults(thinking=False)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))

    screening_thinking = (
        True if args.screening_thinking == "true" else False if args.screening_thinking == "false" else None
    )
    screening_records = load_screening_records_jsonl(
        args.screening_results,
        screening_thinking=screening_thinking,
    )
    pricing_overrides = load_pricing_overrides(args.pricing_json)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    completed_keys = load_completed_understanding_keys(out_path) if args.skip_existing else set()

    with out_path.open("a", encoding="utf-8") as handle:
        for model in args.model:
            pricing = pricing_for_model(model, pricing_overrides)
            for record in iter_understanding_records_from_screening(
                screening_records,
                model=model,
                pricing=pricing,
                base_url=args.base_url,
                api_key=args.api_key,
                thinking=args.thinking,
                existing_keys=completed_keys,
            ):
                handle.write(record.model_dump_json() + "\n")
                handle.flush()
                if record.status.ok:
                    completed_keys.add(understanding_record_key(record))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
