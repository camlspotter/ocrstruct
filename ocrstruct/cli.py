from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ocrstruct.html import elements_to_html
from ocrstruct.pdf import convert_pdf_to_elements
from ocrstruct.types import elements_to_markdown


logger = logging.getLogger(__name__)


def _default_output_dir(pdf_path: Path) -> Path:
    return pdf_path.with_suffix("")


def _write_outputs(outdir: Path, markdown_text: str, html_text: str | None) -> tuple[Path, Path | None]:
    text_md = outdir / "text.md"
    text_md.write_text(markdown_text, encoding="utf-8")
    if html_text is None:
        logger.info("pandoc not found or failed; skip HTML conversion")
        return text_md, None

    text_html = outdir / "text.html"
    text_html.write_text(html_text, encoding="utf-8")
    return text_md, text_html


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="ocrstruct",
        description="Convert PDF to markdown + images using MinerU.",
    )
    parser.add_argument("pdf", nargs="?", help="input PDF path")
    parser.add_argument(
        "--outdir",
        help="output directory (default: <pdf-basename-without-ext>/ or directory containing middle.json/elements.json)",
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
        "--lazy",
        action="store_true",
        help="reuse existing middle.json in the output directory when available",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="log level",
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))

    if not args.pdf:
        raise ValueError("pdf is required")

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    outdir = Path(args.outdir) if args.outdir else _default_output_dir(pdf_path)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "images").mkdir(parents=True, exist_ok=True)

    elements_json_path = outdir / 'elements.json'
    elements = convert_pdf_to_elements(
        str(pdf_path),
        outdir=str(outdir),
        backend=args.backend,
        method=args.method,
        lang=args.lang,
        server_url=args.server_url,
        seal_enable=not args.disable_seal,
        lazy=args.lazy,
    )

    markdown_text = elements_to_markdown(elements, llm=True)
    html_text = elements_to_html(elements)
    text_md, text_html = _write_outputs(
        outdir,
        markdown_text,
        html_text,
    )

    logger.info("Wrote elements: %s", elements_json_path)
    logger.info("Wrote markdown: %s", text_md)
    if text_html is not None:
        logger.info("Wrote html: %s", text_html)
    logger.info("Images dir: %s", outdir / "images")
    logger.info("Middle JSON: %s", outdir / "middle.json")
    print(text_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
