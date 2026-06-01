from __future__ import annotations

from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from ocrstruct.middle import Middle, Result
from ocrstruct.middle_to_markdown import RenderOptions, middle_to_markdown, result_to_markdown


_EQ_TAG_RE = re.compile(r"<eq>(.*?)</eq>", re.IGNORECASE | re.DOTALL)


def default_html_header_path() -> Path | None:
    header = Path(__file__).with_name("style.html")
    if header.exists():
        return header
    return None


def _postprocess_html_mathjax_eq(html: str) -> str:
    '''<eq>..</eq> -> <span class="math inilne">..</span>'''
    if "<eq>" not in html.lower():
        return html

    def repl(match: re.Match[str]) -> str:
        expr = match.group(1).strip()
        return f'<span class="math inline">\\({expr}\\)</span>'

    return _EQ_TAG_RE.sub(repl, html)


def markdown_to_html(
    markdown_text: str,
    *,
    header_path: str | Path | None = None,
) -> str | None:
    pandoc = shutil.which("pandoc")
    if pandoc is None:
        return None

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".md",
        prefix="ocrstruct-html-render-",
        delete=False,
    ) as tmp_md:
        tmp_md.write(markdown_text)
        tmp_md_path = Path(tmp_md.name)

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".html",
        prefix="ocrstruct-html-render-",
        delete=False,
    ) as tmp_html:
        tmp_html_path = Path(tmp_html.name)

    command = [
        pandoc,
        "--from=markdown+tex_math_dollars",
        "--mathjax",
        "--standalone",
        str(tmp_md_path),
        "-o",
        str(tmp_html_path),
    ]
    resolved_header_path = Path(header_path) if header_path is not None else default_html_header_path()
    if resolved_header_path is not None:
        command.extend(["--include-in-header", str(resolved_header_path)])

    try:
        subprocess.run(command, check=True)
        html = tmp_html_path.read_text(encoding="utf-8")
    except subprocess.CalledProcessError:
        return None
    finally:
        tmp_md_path.unlink(missing_ok=True)
        tmp_html_path.unlink(missing_ok=True)

    return _postprocess_html_mathjax_eq(html)


def middle_to_html(
    middle: Middle,
    *,
    header_path: str | Path | None = None,
    options: RenderOptions | None = None,
) -> str | None:
    return markdown_to_html(
        middle_to_markdown(middle, options=options),
        header_path=header_path,
    )


def result_to_html(
    result: Result,
    *,
    header_path: str | Path | None = None,
    options: RenderOptions | None = None,
) -> str | None:
    return markdown_to_html(
        result_to_markdown(result, options=options),
        header_path=header_path,
    )
