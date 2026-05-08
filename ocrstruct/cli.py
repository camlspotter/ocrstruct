from __future__ import annotations

import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv

from ocrstruct.chunk import Chunk, chunk_middle
from ocrstruct.html import result_to_html
from ocrstruct.image_understanding import (
    image_refs_from_middle,
    iter_screening_records_from_refs,
    iter_understanding_records_from_screening,
    load_pricing_overrides,
    load_screening_records_jsonl,
    load_understanding_records_jsonl,
    merge_understanding_into_middle,
    pricing_for_model,
)
from ocrstruct.middle import Result
from ocrstruct.middle_to_markdown import RenderOptions, result_to_markdown
from ocrstruct.pdf import convert_pdf_to_middle
import ocrstruct.utils as utils


logger = logging.getLogger(__name__)


def _default_output_dir(pdf_path: Path) -> Path:
    return pdf_path.with_suffix("")


def _write_outputs(
    outdir: Path, 
    markdown_text: str, 
    html_text: str | None,
    chunks : list[Chunk] | None,
) -> None:
    text_md = outdir / "text.md"
    text_md.write_text(markdown_text, encoding="utf-8")
    logger.info(f"wrote {text_md}")

    if html_text is None:
        logger.info("pandoc not found or failed; skip HTML conversion")
    else:
        text_html = outdir / "text.html"
        text_html.write_text(html_text, encoding="utf-8")
        logger.info(f"wrote {text_html}")

    if chunks:
        chunks_json = outdir / "chunks.json"
        utils.save_json(list[Chunk], chunks_json, chunks)
        logger.info(f"wrote {chunks_json}")


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


def _generate_image_understanding(
    args: argparse.Namespace,
    *,
    outdir: Path,
    pdf_path: Path,
    result: Result,
) -> None:
    if not args.with_image_understanding:
        return

    if not args.image_screening_model:
        raise ValueError("--with-image-understanding requires --image-screening-model")
    if not args.image_understanding_model:
        raise ValueError("--with-image-understanding requires --image-understanding-model")

    middle_json_path = outdir / "middle.json"
    refs = image_refs_from_middle(
        result.middle_json,
        pdf_path=str(pdf_path),
        middle_json_path=str(middle_json_path),
    )
    if not refs:
        logger.info("No image refs found for image understanding")
        return

    pricing_overrides = load_pricing_overrides(args.model_pricing_json)
    screening_path = outdir / "image_screening.jsonl"
    if args.lazy and screening_path.exists():
        logger.info("Reusing image screening results: %s", screening_path)
    else:
        screening_path.parent.mkdir(parents=True, exist_ok=True)
        with screening_path.open("w", encoding="utf-8") as handle:
            for model in args.image_screening_model:
                pricing = pricing_for_model(model, pricing_overrides)
                for record in iter_screening_records_from_refs(
                    refs,
                    model=model,
                    pricing=pricing,
                    base_url=args.image_screening_base_url,
                    api_key=args.image_screening_api_key,
                    thinking=False,
                ):
                    handle.write(record.model_dump_json() + "\n")
                    handle.flush()
        logger.info("Wrote image screening results: %s", screening_path)

    screening_records = load_screening_records_jsonl(screening_path)
    if not screening_records:
        logger.info("No successful screening records found for image understanding")
        return

    understanding_path = outdir / "image_understanding.jsonl"
    if args.lazy and understanding_path.exists():
        logger.info("Reusing image understanding results: %s", understanding_path)
        return

    with understanding_path.open("w", encoding="utf-8") as handle:
        for model in args.image_understanding_model:
            pricing = pricing_for_model(model, pricing_overrides)
            for record in iter_understanding_records_from_screening(
                screening_records,
                model=model,
                pricing=pricing,
                base_url=args.image_understanding_base_url,
                api_key=args.image_understanding_api_key,
                thinking=False,
            ):
                handle.write(record.model_dump_json() + "\n")
                handle.flush()
    logger.info("Wrote image understanding results: %s", understanding_path)


def _convert_one_pdf(args: argparse.Namespace, pdf_path: Path) -> None:
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

    _generate_image_understanding(
        args,
        outdir=outdir,
        pdf_path=pdf_path,
        result=result,
    )
    _merge_image_understanding_if_present(outdir, result=result)
    middle = result.middle_json

    markdown_text = result_to_markdown(result)
    html_text = result_to_html(result)
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
        args.chunk_chars,
        args.chunk_overlap_chars,
    )

    _write_outputs(
        outdir,
        markdown_text,
        html_text,
        chunked.with_overlap,
    )


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
        "--chunk-chars",
        type=int,
        default=800,
        help="target chunk size in characters for chunks.json",
    )
    parser.add_argument(
        "--chunk-overlap-chars",
        type=int,
        default=200,
        help="overlap size in characters between adjacent chunks",
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
        _convert_one_pdf(args, Path(pdf_arg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
