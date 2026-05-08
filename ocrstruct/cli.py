from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ocrstruct.html import result_to_html
from ocrstruct.image_understanding import (
    load_understanding_records_jsonl,
    merge_understanding_into_middle,
)
from ocrstruct.middle import Result
from ocrstruct.middle_to_markdown import result_to_markdown, RenderOptions
from ocrstruct.pdf import convert_pdf_to_middle


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


def _merge_image_understanding_if_present(outdir: Path, *, result: Result) -> None:
    understanding_jsonl_path = outdir / "image_understanding.jsonl"
    if not understanding_jsonl_path.exists():
        return

    understanding_records = load_understanding_records_jsonl(understanding_jsonl_path)
    result.middle_json = merge_understanding_into_middle(
        result.middle_json,
        understanding_records,
    )
    logger.info("Merged image understanding: %s", understanding_jsonl_path)


def _convert_one_pdf(args: argparse.Namespace, pdf_path: Path) -> Path:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    outdir = Path(args.outdir) if args.outdir else _default_output_dir(pdf_path)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "images").mkdir(parents=True, exist_ok=True)

    result = convert_pdf_to_middle(
        str(pdf_path),
        outdir=str(outdir),
        backend=args.backend,
        method=args.method,
        lang=args.lang,
        server_url=args.server_url,
        seal_enable=not args.disable_seal,
        formula_enable=not args.disable_formula,
        lazy=args.lazy,
    )

    _merge_image_understanding_if_present(outdir, result=result)

    middle = result.middle_json
    from ocrstruct.chunk import chunk_middle
    chunked = chunk_middle(
        middle, 
        RenderOptions(
            table_multicell_mode= 'repeat', 
            image_understanding_render_mode= 'long',
            include_source_image_links= False,
            render_latex_as_unicode_text= True,
            include_images= False,
            include_image_understanding= "rag",
        ),  
        800,
        1200,
    )
    for c in chunked.without_overlap:
        print(c.str)
        print('======')

    markdown_text = result_to_markdown(result)
    html_text = result_to_html(result)
    text_md, text_html = _write_outputs(
        outdir,
        markdown_text,
        html_text,
    )

    logger.info("Wrote markdown: %s", text_md)
    if text_html is not None:
        logger.info("Wrote html: %s", text_html)
    logger.info("Images dir: %s", outdir / "images")
    logger.info("Middle JSON: %s", outdir / "middle.json")
    return text_md


def main() -> int:
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
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="log level",
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))

    if len(args.pdf) > 1 and args.outdir:
        raise ValueError("--outdir cannot be used with multiple PDF inputs")

    text_mds = [
        _convert_one_pdf(args, Path(pdf_arg))
        for pdf_arg in args.pdf
    ]
    for text_md in text_mds:
        print(text_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
