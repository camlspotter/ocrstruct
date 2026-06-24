from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Sequence
from pathlib import Path

from dotenv import load_dotenv

from ocrstruct.api import convert_one_pdf, render_middle


logger = logging.getLogger(__name__)


def _find_pdfs(data_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in data_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".pdf"
    )


def _env_flag(name: str, *, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_model(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return None
    return value


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Convert PDF fixtures under tests/data with ocrstruct.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).with_name("data"),
        help="directory containing source PDFs",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="maximum number of PDFs to process",
    )
    parser.add_argument(
        "--disable-seal",
        action="store_true",
        default=True,
        help="skip MinerU seal OCR prediction",
    )
    parser.add_argument(
        "--enable-seal",
        action="store_false",
        dest="disable_seal",
        help="enable MinerU seal OCR prediction",
    )
    parser.add_argument(
        "--disable-formula",
        action="store_true",
        help="skip MinerU formula recognition",
    )
    parser.add_argument(
        "--with-image-understanding",
        action="store_true",
        default=_env_flag("OCRSTRUCT_WITH_IMAGE_UNDERSTANDING"),
        help="also run VLM-based image screening and image understanding",
    )
    parser.add_argument(
        "--image-screening-model",
        default=_env_model("OCRSTRUCT_IMAGE_SCREENING_MODEL"),
        help="screening model to use when --with-image-understanding is enabled",
    )
    parser.add_argument(
        "--image-screening-base-url",
        default=os.environ.get("OCRSTRUCT_IMAGE_SCREENING_BASE_URL"),
        help="OpenAI-compatible base URL for image screening",
    )
    parser.add_argument(
        "--image-screening-api-key",
        default=os.environ.get("OCRSTRUCT_IMAGE_SCREENING_API_KEY"),
        help="API key for image screening; defaults to values from .env",
    )
    parser.add_argument(
        "--image-understanding-model",
        default=_env_model("OCRSTRUCT_IMAGE_UNDERSTANDING_MODEL"),
        help="understanding model to use when --with-image-understanding is enabled",
    )
    parser.add_argument(
        "--image-understanding-base-url",
        default=os.environ.get("OCRSTRUCT_IMAGE_UNDERSTANDING_BASE_URL"),
        help="OpenAI-compatible base URL for image understanding",
    )
    parser.add_argument(
        "--image-understanding-api-key",
        default=os.environ.get("OCRSTRUCT_IMAGE_UNDERSTANDING_API_KEY"),
        help="API key for image understanding; defaults to values from .env",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="continue processing after a per-file failure",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="log level",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=getattr(logging, args.log_level))

    data_dir = args.data_dir.resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"data directory not found: {data_dir}")

    pdf_paths = _find_pdfs(data_dir)
    if args.limit is not None:
        pdf_paths = pdf_paths[: args.limit]

    logger.info("Found %s PDF fixtures in %s", len(pdf_paths), data_dir)
    failures: list[Path] = []

    for index, pdf_path in enumerate(pdf_paths, start=1):
        outdir = pdf_path.with_suffix("")
        logger.info("----------------------------------------------------------")
        logger.info("[%s/%s] converting %s", index, len(pdf_paths), pdf_path.name)
        try:
            result = convert_one_pdf(
                pdf_path=pdf_path,
                outdir=outdir,
                seal_enable=not args.disable_seal,
                formula_enable=not args.disable_formula,
                with_image_understanding=args.with_image_understanding,
                image_screening_model=args.image_screening_model,
                image_screening_base_url=args.image_screening_base_url,
                image_screening_api_key=args.image_screening_api_key,
                image_understanding_model=args.image_understanding_model,
                image_understanding_base_url=args.image_understanding_base_url,
                image_understanding_api_key=args.image_understanding_api_key,
            )
            render_middle(outdir, result.middle)
        except Exception:
            failures.append(pdf_path)
            logger.exception("Failed to convert %s", pdf_path)
            if not args.keep_going:
                break

    if failures:
        logger.error(
            "Finished with %s failure(s): %s",
            len(failures),
            ", ".join(path.name for path in failures),
        )
        return 1

    logger.info("Finished converting %s PDF fixture(s)", len(pdf_paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
