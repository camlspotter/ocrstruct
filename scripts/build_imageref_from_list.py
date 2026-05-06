from __future__ import annotations

import argparse
import html
import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, cast

if TYPE_CHECKING:
    from ocrstruct.image_understanding import ImageRef
    from ocrstruct.middle import Result


REPO_ROOT = Path(__file__).resolve().parents[1]
OCRSTRUCT_DIR = REPO_ROOT / "ocrstruct"
DEFAULT_ROOT = Path("~/rois-rag/_data/staff_page2").expanduser().resolve()


def _load_ocrstruct_modules() -> tuple[type["Result"], Callable[..., list["ImageRef"]]]:
    pkg = types.ModuleType("ocrstruct")
    pkg.__path__ = [str(OCRSTRUCT_DIR)]
    sys.modules["ocrstruct"] = pkg

    loaded: dict[str, object] = {}
    for name in ("utils", "middle", "image_understanding"):
        path = OCRSTRUCT_DIR / f"{name}.py"
        spec = importlib.util.spec_from_file_location(f"ocrstruct.{name}", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load module: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"ocrstruct.{name}"] = module
        spec.loader.exec_module(module)
        loaded[name] = module

    middle_mod = cast(Any, loaded["middle"])
    image_mod = cast(Any, loaded["image_understanding"])
    return middle_mod.Result, image_mod.image_refs_from_middle


def _iter_middle_jsons(root: Path) -> list[Path]:
    return sorted(root.rglob("middle.json"))


def _image_file_path(middle_json_path: Path, image_path: str) -> Path:
    path = Path(image_path)
    if path.is_absolute():
        return path
    return middle_json_path.parent / "images" / path


def _pdf_path_from_middle_json(middle_json_path: Path) -> Path:
    parent = middle_json_path.parent
    if parent.parent.name == "__data":
        return parent.parent.parent / parent.name
    return parent


def _load_image_list(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _build_ref_map(root: Path) -> dict[str, dict[str, object]]:
    Result, image_refs_from_middle = _load_ocrstruct_modules()
    ref_map: dict[str, dict[str, object]] = {}
    for middle_json_path in _iter_middle_jsons(root):
        result = Result.model_validate_json(middle_json_path.read_text())
        refs = image_refs_from_middle(
            result.middle_json,
            pdf_path=str(_pdf_path_from_middle_json(middle_json_path)),
            middle_json_path=str(middle_json_path),
        )
        for ref in refs:
            full_path = str(_image_file_path(Path(ref.middle_json_path), ref.image_path))
            ref_map[full_path] = ref.model_dump(mode="json")
    return ref_map


def _ordered_refs(image_paths: list[str], ref_map: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    missing = [path for path in image_paths if path not in ref_map]
    if missing:
        missing_text = "\n".join(missing[:20])
        raise ValueError(
            f"Could not find {len(missing)} image paths in middle.json data.\n{missing_text}"
        )
    return [ref_map[path] for path in image_paths]


def _json_output_path(list_path: Path) -> Path:
    return list_path.with_suffix(".imageref.json")


def _html_output_path(list_path: Path) -> Path:
    return list_path.with_suffix(".imageref.html")


def _text_value(value: object | None) -> str:
    if value is None:
        return ""
    return html.escape(str(value)).replace("\n", "<br>")


def _page_number_from_item(item: dict[str, object]) -> int:
    page_idx_value = item.get("page_idx")
    if isinstance(page_idx_value, int):
        return page_idx_value + 1
    return 1


def _render_html(items: list[dict[str, object]], title: str) -> str:
    parts: list[str] = [
        "<!doctype html>",
        '<html lang="ja">',
        "<head>",
        '  <meta charset="utf-8">',
        '  <meta name="viewport" content="width=device-width, initial-scale=1">',
        f"  <title>{html.escape(title)}</title>",
        "  <style>",
        "    body { font-family: sans-serif; line-height: 1.4; margin: 24px; }",
        "    .item { margin-bottom: 32px; padding-bottom: 24px; border-bottom: 1px solid #ccc; }",
        "    img { display: block; max-width: 50vw; height: auto; margin: 12px 0; }",
        "    .field { margin: 4px 0; }",
        "    .label { font-weight: 700; }",
        "    a { word-break: break-all; }",
        "  </style>",
        "</head>",
        "<body>",
        f"  <h1>{html.escape(title)} ({len(items)} items)</h1>",
    ]

    field_order = [
        "pdf_path",
        "middle_json_path",
        "page_idx",
        "block_index",
        "block_type",
        "image_path",
        "caption",
        "nearby_text_before",
        "nearby_text_after",
        "section_title",
    ]

    for index, item in enumerate(items, start=1):
        pdf_path = str(item.get("pdf_path", ""))
        middle_json_path = str(item.get("middle_json_path", ""))
        image_path_value = str(item.get("image_path", ""))
        image_src = image_path_value
        if image_path_value and not Path(image_path_value).is_absolute() and middle_json_path:
            image_src = str(Path(middle_json_path).parent / "images" / image_path_value)

        parts.append('  <div class="item">')
        parts.append(f"    <h2>{index}</h2>")
        for field in field_order:
            label = html.escape(field)
            value = item.get(field)
            if field == "pdf_path":
                page_number = _page_number_from_item(item)
                href = html.escape(f"{pdf_path}#page={page_number}", quote=True)
                text = html.escape(pdf_path)
                parts.append(
                    f'    <div class="field"><span class="label">{label}:</span> '
                    f'<a href="{href}" target="_blank" rel="noopener noreferrer">{text}</a></div>'
                )
                continue
            if field == "middle_json_path":
                href = html.escape(middle_json_path, quote=True)
                text = html.escape(middle_json_path)
                parts.append(
                    f'    <div class="field"><span class="label">{label}:</span> '
                    f'<a href="{href}" target="_blank" rel="noopener noreferrer">{text}</a></div>'
                )
                continue
            if field == "image_path":
                src = html.escape(image_src, quote=True)
                alt = html.escape(image_path_value)
                text = html.escape(image_path_value)
                parts.append(
                    f'    <div class="field"><span class="label">{label}:</span> {text}</div>'
                )
                parts.append(f'    <img src="{src}" alt="{alt}">')
                continue
            parts.append(
                f'    <div class="field"><span class="label">{label}:</span> '
                f"{_text_value(value)}</div>"
            )
        parts.append("  </div>")

    parts.extend(["</body>", "</html>"])
    return "\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image_list", help="Text file with one absolute image path per line.")
    parser.add_argument(
        "--root",
        default=str(DEFAULT_ROOT),
        help="Directory tree to search for middle.json",
    )
    args = parser.parse_args()

    list_path = Path(args.image_list).expanduser().resolve()
    root = Path(args.root).expanduser().resolve()
    image_paths = _load_image_list(list_path)
    ref_map = _build_ref_map(root)
    items = _ordered_refs(image_paths, ref_map)

    json_path = _json_output_path(list_path)
    html_path = _html_output_path(list_path)
    json_path.write_text(json.dumps(items, ensure_ascii=False, indent=2) + "\n")
    html_path.write_text(_render_html(items, list_path.stem))

    print(json_path)
    print(html_path)


if __name__ == "__main__":
    main()
