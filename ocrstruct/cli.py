from __future__ import annotations

import argparse
import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
import os

from ocrstruct.pdf import convert_pdf_to_elements
from ocrstruct.types import elements_to_markdown, Element
from ocrstruct.middle_to_elements import middle_to_elements
from ocrstruct.table import decode_html_table_eq_tokens
from ocrstruct.utils import load_json, save_json


logger = logging.getLogger(__name__)
_EQ_TAG_RE = re.compile(r"<eq>(.*?)</eq>", re.IGNORECASE | re.DOTALL)


def _default_output_dir(pdf_path: Path) -> Path:
    return pdf_path.with_suffix("")


def _default_middle_output_dir(middle_json_path: Path) -> Path:
    return middle_json_path.parent


def _default_elements_output_dir(elements_json_path: Path) -> Path:
    return elements_json_path.parent


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
    _postprocess_html_mathjax_eq(text_html)
    return text_html


def _convert_markdown_string_to_html_if_pandoc_exists(markdown_text: str, text_html: Path) -> Path | None:
    pandoc = shutil.which("pandoc")
    if pandoc is None:
        logger.info("pandoc not found; skip HTML conversion")
        return None

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".md",
        prefix="ocrstruct-html-render-",
        delete=False,
    ) as tmp:
        tmp.write(markdown_text)
        tmp_md = Path(tmp.name)

    command = [
        pandoc,
        "--from=markdown+tex_math_dollars",
        "--mathjax",
        "--standalone",
        str(tmp_md),
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
    finally:
        tmp_md.unlink(missing_ok=True)
    _postprocess_html_mathjax_eq(text_html)
    return text_html


def _postprocess_html_mathjax_eq(text_html: Path) -> None:
    html = text_html.read_text(encoding="utf-8")
    if "<eq>" not in html.lower() and "CODEXEQ[" not in html:
        return

    def repl(m: re.Match[str]) -> str:
        expr = m.group(1).strip()
        return f'<span class="math inline">\\({expr}\\)</span>'

    updated = _EQ_TAG_RE.sub(repl, html)
    updated = decode_html_table_eq_tokens(updated)
    if updated != html:
        text_html.write_text(updated, encoding="utf-8")


def _write_outputs(
    outdir: Path,
    markdown_text: str | None,
    *,
    html_markdown_text: str | None = None,
) -> tuple[Path, Path | None]:
    text_md = outdir / "text.md"
    text_md.write_text(markdown_text or "", encoding="utf-8")
    if html_markdown_text is None:
        text_html = _convert_markdown_to_html_if_pandoc_exists(text_md)
    else:
        text_html = _convert_markdown_string_to_html_if_pandoc_exists(
            html_markdown_text,
            outdir / "text.html",
        )
    return text_md, text_html


def _write_elements_json(outdir: Path, elements_json_path: Path | None, elements: list[Element]) -> None:
    target = elements_json_path or (outdir / "elements.json")
    save_json(list[Element], target, elements)


def _validate_from_middle_args(args: argparse.Namespace) -> None:
    ignored_args = {
        "--backend": args.backend,
        "--method": args.method,
        "--lang": args.lang,
        "--server-url": args.server_url,
        "--disable-seal": args.disable_seal if args.disable_seal else None,
        "--lazy": args.lazy if args.lazy else None,
    }
    used = [name for name, value in ignored_args.items() if value is not None]
    if used:
        raise ValueError(f"{', '.join(used)} cannot be used with --from-middle")


def _validate_from_elements_args(args: argparse.Namespace) -> None:
    ignored_args = {
        "--backend": args.backend,
        "--method": args.method,
        "--lang": args.lang,
        "--server-url": args.server_url,
        "--from-middle": args.from_middle,
        "--disable-seal": args.disable_seal if args.disable_seal else None,
        "--lazy": args.lazy if args.lazy else None,
    }
    used = [name for name, value in ignored_args.items() if value is not None]
    if used:
        raise ValueError(f"{', '.join(used)} cannot be used with --from-elements")


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

    text_md, text_html = _write_outputs(
        outdir,
        elements_to_markdown(elements, llm=True),
        html_markdown_text=elements_to_markdown(elements, llm=False),
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
