from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
import tempfile
import types
from os.path import relpath
from pathlib import Path


REPO_ROOT = Path("/Users/jun/mocrdown")
DATA_ROOT = Path("/Users/jun/rois-rag/_data/staff_page2")
STYLE_HTML = REPO_ROOT / "ocrstruct" / "style.html"


def load_modules() -> None:
    pkg = types.ModuleType("ocrstruct")
    pkg.__path__ = [str(REPO_ROOT / "ocrstruct")]
    sys.modules["ocrstruct"] = pkg

    for name in ["utils", "middle", "table", "middle_to_markdown"]:
        path = REPO_ROOT / "ocrstruct" / f"{name}.py"
        spec = importlib.util.spec_from_file_location(f"ocrstruct.{name}", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to load module spec for {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"ocrstruct.{name}"] = module
        spec.loader.exec_module(module)


def markdown_to_html(markdown_text: str) -> str:
    pandoc = shutil.which("pandoc")
    if pandoc is None:
        raise RuntimeError("pandoc not found")

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".md",
        prefix="middle-render-",
        delete=False,
    ) as tmp_md:
        tmp_md.write(markdown_text)
        tmp_md_path = Path(tmp_md.name)

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".html",
        prefix="middle-render-",
        delete=False,
    ) as tmp_html:
        tmp_html_path = Path(tmp_html.name)

    command = [
        pandoc,
        "--from=markdown+raw_html+tex_math_dollars",
        "--mathjax",
        "--standalone",
        str(tmp_md_path),
        "-o",
        str(tmp_html_path),
    ]
    if STYLE_HTML.exists():
        command.extend(["--include-in-header", str(STYLE_HTML)])

    try:
        subprocess.run(command, check=True)
        return tmp_html_path.read_text(encoding="utf-8")
    finally:
        tmp_md_path.unlink(missing_ok=True)
        tmp_html_path.unlink(missing_ok=True)


def prepend_original_pdf_link(markdown_text: str, *, middle_json_path: Path) -> str:
    pdf_dir = middle_json_path.parent
    original_pdf_path = middle_json_path.parents[2] / pdf_dir.name
    relative_pdf_path = relpath(original_pdf_path, start=pdf_dir)
    link_html = f'<p><a href="{relative_pdf_path}">Original PDF</a></p>'
    if not markdown_text.strip():
        return link_html
    return f"{link_html}\n\n{markdown_text}"


def main() -> int:
    load_modules()

    from ocrstruct.middle import Result, merge_discarded_blocks
    from ocrstruct.middle_to_markdown import result_to_markdown

    paths = sorted(DATA_ROOT.glob("**/middle.json"))
    failures: list[tuple[Path, Exception]] = []

    for path in paths:
        try:
            result = Result.model_validate_json(path.read_text(encoding="utf-8"))
            restored_middle = merge_discarded_blocks(result.middle_json)
            restored_result = result.model_copy(update={"middle_json": restored_middle})
            markdown_text = result_to_markdown(restored_result)
            markdown_text = prepend_original_pdf_link(markdown_text, middle_json_path=path)
            html_text = markdown_to_html(markdown_text)
            out_path = path.with_name("middle.html")
            out_path.write_text(html_text, encoding="utf-8")
        except Exception as exc:
            failures.append((path, exc))

    print(f"files={len(paths)} failed={len(failures)}")
    for path, exc in failures[:10]:
        print("---")
        print(path)
        print(type(exc).__name__)
        print(str(exc)[:4000])

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
