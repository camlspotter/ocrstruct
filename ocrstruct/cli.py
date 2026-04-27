from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
from pathlib import Path

from ocrstruct.pdf import convert_pdf_to_middle_and_markdown, load_middle_json, render_middle_json_to_markdown


logger = logging.getLogger(__name__)


def _default_output_dir(pdf_path: Path) -> Path:
    return pdf_path.with_suffix("")


def _default_middle_output_dir(middle_json_path: Path) -> Path:
    return middle_json_path.parent


def _default_html_header() -> Path | None:
    header = Path(__file__).with_name("style.html")
    if header.exists():
        return header
    return None


def _convert_markdown_to_html_if_pandoc_exists(text_md: Path) -> Path | None:
    pandoc = shutil.which("pandoc")
    if pandoc is None:
        logger.info("pandoc not found; skip HTML conversion")
        return None

    text_html = text_md.with_name("text.html")
    command = [
        pandoc,
        "--from=markdown+tex_math_dollars",
        "--mathjax",
        "--standalone",
        str(text_md),
        "-o",
        str(text_html),
    ]
    header_include = _default_html_header()
    if header_include is not None:
        command.extend(["--include-in-header", str(header_include)])

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        logger.warning("pandoc failed (exit=%s); skip HTML output", e.returncode)
        return None
    return text_html


def _write_outputs(outdir: Path, markdown_text: str | None) -> tuple[Path, Path | None]:
    text_md = outdir / "text.md"
    text_md.write_text(markdown_text or "", encoding="utf-8")
    text_html = _convert_markdown_to_html_if_pandoc_exists(text_md)
    return text_md, text_html


def _validate_from_middle_args(args: argparse.Namespace) -> None:
    ignored_args = {
        "--backend": args.backend,
        "--method": args.method,
        "--lang": args.lang,
        "--server-url": args.server_url,
    }
    used = [name for name, value in ignored_args.items() if value is not None]
    if used:
        raise ValueError(f"{', '.join(used)} cannot be used with --from-middle")


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="ocrstruct",
        description="Convert PDF to markdown + images using MinerU.",
    )
    parser.add_argument("pdf", nargs="?", help="input PDF path")
    parser.add_argument(
        "--from-middle",
        help="reuse an existing middle.json and render text.md/text.html without re-running OCR",
    )
    parser.add_argument(
        "--outdir",
        help="output directory (default: <pdf-basename-without-ext>/ or directory containing middle.json)",
    )
    parser.add_argument("--backend", help="MINERU_BACKEND override")
    parser.add_argument("--method", help="MINERU_METHOD override")
    parser.add_argument("--lang", help="MINERU_LANG override")
    parser.add_argument("--server-url", help="MINERU_SERVER_URL override")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="log level",
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))

    if args.from_middle:
        _validate_from_middle_args(args)
        middle_json_path = Path(args.from_middle).expanduser().resolve()
        if not middle_json_path.exists():
            raise FileNotFoundError(f"middle.json not found: {middle_json_path}")
        outdir = Path(args.outdir).expanduser().resolve() if args.outdir else _default_middle_output_dir(middle_json_path)
        outdir.mkdir(parents=True, exist_ok=True)
        result = render_middle_json_to_markdown(
            load_middle_json(middle_json_path),
            markdown_image_bucket_path="images",
        )
        text_md, text_html = _write_outputs(outdir, result.markdown_text)
        logger.info("Rendered markdown from middle JSON: %s", middle_json_path)
        logger.info("Markdown renderer: %s", result.extracted_by)
        logger.info("Wrote markdown: %s", text_md)
        if text_html is not None:
            logger.info("Wrote html: %s", text_html)
        print(text_md)
        return 0

    if not args.pdf:
        raise ValueError("pdf is required unless --from-middle is used")

    pdf_path = Path(args.pdf).expanduser().resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else _default_output_dir(pdf_path)
    images_dir = outdir / "images"
    outdir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    result = convert_pdf_to_middle_and_markdown(
        str(pdf_path),
        tmpdir=str(outdir),
        image_dir=str(images_dir),
        markdown_image_bucket_path="images",
        backend=args.backend,
        method=args.method,
        lang=args.lang,
        server_url=args.server_url,
    )

    text_md, text_html = _write_outputs(outdir, result.markdown_text)

    logger.info("Wrote markdown: %s", text_md)
    if text_html is not None:
        logger.info("Wrote html: %s", text_html)
    logger.info("Images dir: %s", images_dir)
    logger.info("Middle JSON: %s", outdir / "middle.json")
    print(text_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
