from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from pydantic import ValidationError

from ocrstruct.image_understanding import (
    DEFAULT_MODEL_PRICING,
    ImageRef,
    Model,
    ModelPricing,
    ScreeningRunResult,
    screening_run_from_image_ref,
)


class EvalStatus(Model):
    ok: bool
    error: str | None = None


class EvalRecord(Model):
    ref: ImageRef
    model: str
    base_url: str | None = None
    latency_sec: float
    status: EvalStatus
    run: ScreeningRunResult | None = None


def _load_eval_set(path: Path) -> list[ImageRef]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return [ImageRef.model_validate(item) for item in data]
    if isinstance(data, dict) and "items" in data:
        items = data["items"]
        if isinstance(items, list):
            return [ImageRef.model_validate(item) for item in items]
    raise ValueError(f"Unsupported eval set format: {path}")


def _load_pricing_overrides(path: Path | None) -> dict[str, ModelPricing]:
    if path is None:
        return {}
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Pricing override must be a JSON object: {path}")
    out: dict[str, ModelPricing] = {}
    for model, value in data.items():
        out[model] = ModelPricing.model_validate(value)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--base-url")
    parser.add_argument("--api-key")
    parser.add_argument("--pricing-json")
    args = parser.parse_args()

    eval_set_path = Path(args.eval_set)
    out_path = Path(args.out)
    pricing_overrides = _load_pricing_overrides(
        Path(args.pricing_json) if args.pricing_json is not None else None
    )
    refs = _load_eval_set(eval_set_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("a") as f:
        for model in args.model:
            pricing = pricing_overrides.get(model, DEFAULT_MODEL_PRICING.get(model))
            for ref in refs:
                started = time.perf_counter()
                try:
                    run = screening_run_from_image_ref(
                        ref,
                        model=model,
                        base_url=args.base_url,
                        api_key=args.api_key,
                        pricing=pricing,
                    )
                    record = EvalRecord(
                        ref=ref,
                        model=model,
                        base_url=args.base_url,
                        latency_sec=time.perf_counter() - started,
                        status=EvalStatus(ok=True),
                        run=run,
                    )
                except (OSError, ValueError, ValidationError) as e:
                    record = EvalRecord(
                        ref=ref,
                        model=model,
                        base_url=args.base_url,
                        latency_sec=time.perf_counter() - started,
                        status=EvalStatus(ok=False, error=str(e)),
                        run=None,
                    )
                f.write(record.model_dump_json() + "\n")
                f.flush()


if __name__ == "__main__":
    main()
