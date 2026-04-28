from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import types
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, cast

if TYPE_CHECKING:
    from ocrstruct.image_understanding import ImageRef
    from ocrstruct.middle import Result


REPO_ROOT = Path(__file__).resolve().parents[1]
OCRSTRUCT_DIR = REPO_ROOT / "ocrstruct"


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="~/rois-rag/_data/staff_page2",
        help="Directory tree to search for middle.json",
    )
    parser.add_argument(
        "--out",
        help="Optional output file. If omitted, prints to stdout.",
    )
    parser.add_argument(
        "--format",
        choices=("txt", "json"),
        default="txt",
        help="Output format.",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    Result, image_refs_from_middle = _load_ocrstruct_modules()

    refs: list[ImageRef] = []
    for middle_json_path in _iter_middle_jsons(root):
        result = Result.model_validate_json(middle_json_path.read_text())
        refs.extend(
            image_refs_from_middle(
                result.middle_json,
                pdf_path=str(middle_json_path.parent),
                middle_json_path=str(middle_json_path),
            )
        )

    if args.format == "txt":
        lines = [
            str(_image_file_path(Path(ref.middle_json_path), ref.image_path))
            for ref in refs
        ]
        output = "\n".join(lines) + ("\n" if lines else "")
    else:
        items = [
            {
                "pdf_path": ref.pdf_path,
                "middle_json_path": ref.middle_json_path,
                "page_idx": ref.page_idx,
                "block_index": ref.block_index,
                "block_type": ref.block_type,
                "image_path": ref.image_path,
                "image_file_path": str(
                    _image_file_path(Path(ref.middle_json_path), ref.image_path)
                ),
                "caption": ref.caption,
                "nearby_text_before": ref.nearby_text_before,
                "nearby_text_after": ref.nearby_text_after,
                "section_title": ref.section_title,
            }
            for ref in refs
        ]
        output = json.dumps(items, ensure_ascii=False, indent=2) + "\n"

    if args.out:
        Path(args.out).expanduser().resolve().write_text(output)
    else:
        try:
            sys.stdout.write(output)
        except BrokenPipeError:
            os._exit(0)


if __name__ == "__main__":
    main()
