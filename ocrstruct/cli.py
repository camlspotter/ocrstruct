from __future__ import annotations

import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv

from ocrstruct.api import convert_one_pdf, render_middle


logger = logging.getLogger(__name__)

def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(
        prog="ocrstruct",
        description="Convert PDF to middle.json, markdown, and HTML using MinerU.",
    )
    parser.add_argument("pdf", nargs="+", help="input PDF path(s)")
    parser.add_argument(
        "--outdir",
        help="output directory (default: <pdf-basename-without-ext>/)",
    )
    parser.add_argument("--backend", help="MINERU_BACKEND override")
    parser.add_argument("--method", help="MINERU_METHOD override")
    parser.add_argument("--lang", help="MINERU_LANG override")
    parser.add_argument("--server-url", help="MINERU_SERVER_URL override")
    parser.add_argument(
        "--disable-seal",
        action="store_true",
        help="skip MinerU seal OCR prediction when supported",
    )
    parser.add_argument(
        "--disable-formula",
        action="store_true",
        help="skip MinerU formula recognition when supported",
    )
    parser.add_argument(
        "--lazy",
        action="store_true",
        help="reuse existing middle.json in the output directory when available",
    )
    parser.add_argument(
        "--with-image-understanding",
        action="store_true",
        help="also generate image_screening.jsonl and image_understanding.jsonl",
    )

    parser.add_argument(
        "--image-screening-model",
        action="append",
        help="screening model to use for image understanding generation",
    )
    parser.add_argument(
        "--image-screening-base-url",
        help="OpenAI-compatible base URL for image screening generation",
    )
    parser.add_argument(
        "--image-screening-api-key",
        help="API key for image screening generation",
    )

    parser.add_argument(
        "--image-understanding-model",
        action="append",
        help="understanding model to use for image understanding generation",
    )
    parser.add_argument(
        "--image-understanding-base-url",
        help="OpenAI-compatible base URL for image understanding generation",
    )
    parser.add_argument(
        "--image-understanding-api-key",
        help="API key for image understanding generation",
    )

    parser.add_argument(
        "--model-pricing-json",
        help="JSON file with model pricing overrides for screening and understanding generation",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="log level",
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))

    if len(args.pdf) > 1 and args.outdir:
        raise ValueError("--outdir cannot be used with multiple PDF inputs")

    for pdf_arg in args.pdf:
        pdf_path = Path(pdf_arg)
        outdir = Path(args.outdir) if args.outdir else pdf_path.with_suffix("")
        result = convert_one_pdf(
            pdf_path=Path(pdf_arg),
            outdir=args.outdir,
            backend=args.backend,
            method=args.method,
            lang=args.lang,
            server_url=args.server_url,
            seal_enable=not args.disable_seal,
            formula_enable=not args.disable_formula,
            lazy=args.lazy,
            with_image_understanding=args.with_image_understanding,
            image_screening_model=args.image_screening_model,
            image_screening_base_url=args.image_screening_base_url,
            image_screening_api_key=args.image_screening_api_key,
            image_understanding_model=args.image_understanding_model,
            image_understanding_base_url=args.image_understanding_base_url,
            image_understanding_api_key=args.image_understanding_api_key,
            model_pricing_json=args.model_pricing_json,
        )
        render_middle(outdir, result.middle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
