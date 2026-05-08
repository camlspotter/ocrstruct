from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ocrstruct.html import middle_to_html
from ocrstruct.image_understanding import (
    load_understanding_records_jsonl,
    merge_understanding_into_middle,
)
from ocrstruct.middle import Result
from ocrstruct.middle_to_markdown import RenderOptions, middle_to_markdown


logger = logging.getLogger(__name__)


def _write_outputs(
    outdir: Path,
    markdown_text: str,
    html_text: str | None,
) -> tuple[Path, Path | None]:
    text_md = outdir / "text.md"
    text_md.write_text(markdown_text, encoding="utf-8")
    if html_text is None:
        logger.info("pandoc not found or failed; skip HTML conversion")
        return text_md, None

    text_html = outdir / "text.html"
    text_html.write_text(html_text, encoding="utf-8")
    return text_md, text_html


def _build_render_options(args: argparse.Namespace) -> RenderOptions:
    return RenderOptions(
        include_image_understanding="html",
        image_understanding_render_mode=args.image_understanding_render_mode,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="ocrstruct-render",
        description="Render text.md/text.html from middle.json and image_understanding.jsonl.",
    )
    parser.add_argument("--middle-json", required=True, help="path to middle.json")
    parser.add_argument(
        "--image-understanding-jsonl",
        required=True,
        help="path to image_understanding.jsonl",
    )
    parser.add_argument(
        "--outdir",
        help="output directory (default: directory containing middle.json)",
    )
    parser.add_argument(
        "--image-understanding-render-mode",
        default="long",
        choices=["short", "long"],
        help="which understanding description to render",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="log level",
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))

    middle_json_path = Path(args.middle_json)
    understanding_jsonl_path = Path(args.image_understanding_jsonl)
    outdir = Path(args.outdir) if args.outdir else middle_json_path.parent

    result = Result.load_json(middle_json_path)
    understanding_records = load_understanding_records_jsonl(understanding_jsonl_path)
    merged_middle = merge_understanding_into_middle(
        result.middle_json,
        understanding_records,
    )
    options = _build_render_options(args)
    markdown_text = middle_to_markdown(merged_middle, options=options)
    html_text = middle_to_html(merged_middle, options=options)

    outdir.mkdir(parents=True, exist_ok=True)
    text_md, text_html = _write_outputs(outdir, markdown_text, html_text)
    logger.info("Wrote markdown: %s", text_md)
    if text_html is not None:
        logger.info("Wrote html: %s", text_html)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
