from __future__ import annotations

import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv

from ocrstruct.result import Result
from ocrstruct.image_understanding import (
    iter_screening_records_from_refs,
    load_completed_screening_keys,
    load_pricing_overrides,
    pricing_for_model,
    image_refs_from_middle,
)

def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(
        prog="ocrstruct-screening",
        description="Generate image screening JSONL directly from a middle.json file.",
    )
    parser.add_argument("--middle-json", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--pdf-path", required=True)
    parser.add_argument("--base-url")
    parser.add_argument("--api-key")
    parser.add_argument("--pricing-json")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip (model, image) pairs already recorded as ok in the output jsonl.",
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

    result = Result.load_json(args.middle_json)
    refs = image_refs_from_middle(result.middle, pdf_path= args.pdf_path, middle_json_path= args.middle_json)
    pricing_overrides = load_pricing_overrides(args.pricing_json)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    completed_keys = load_completed_screening_keys(out_path) if args.skip_existing else set()

    with out_path.open("a", encoding="utf-8") as handle:
        for model in args.model:
            pricing = pricing_for_model(model, pricing_overrides)
            for record in iter_screening_records_from_refs(
                refs,
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
                    completed_keys.add((record.model, record.thinking, (
                        record.ref.middle_json_path,
                        record.ref.page_idx,
                        record.ref.block_index,
                        record.ref.image_path,
                    )))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
