from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
from pathlib import Path

from ocrstruct.pdf import convert_pdf_to_middle_and_markdown


logger = logging.getLogger(__name__)


def _default_output_dir(pdf_path: Path) -> Path:
    return pdf_path.with_suffix("")


def _convert_markdown_to_html_if_pandoc_exists(text_md: Path) -> Path | None:
    pandoc = shutil.which("pandoc")
    if pandoc is None:
        logger.info("pandoc not found; skip HTML conversion")
        return None

    text_html = text_md.with_name("text.html")
    try:
        subprocess.run(
            [
                pandoc,
                "--from=markdown+tex_math_dollars",
                "--mathjax",
                "--standalone",
                str(text_md),
                "-o",
                str(text_html),
            ],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        logger.warning("pandoc failed (exit=%s); skip HTML output", e.returncode)
        return None
    return text_html


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="ocrstruct",
        description="Convert PDF to markdown + images using MinerU.",
    )
    parser.add_argument("pdf", help="input PDF path")
    parser.add_argument(
        "--outdir",
        help="output directory (default: <pdf-basename-without-ext>/)",
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

    text_md = outdir / "text.md"
    text_md.write_text(result.markdown_text or "", encoding="utf-8")
    text_html = _convert_markdown_to_html_if_pandoc_exists(text_md)

    logger.info("Wrote markdown: %s", text_md)
    if text_html is not None:
        logger.info("Wrote html: %s", text_html)
    logger.info("Images dir: %s", images_dir)
    logger.info("Middle JSON: %s", outdir / "middle.json")
    print(text_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
