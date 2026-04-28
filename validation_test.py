from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


REPO_ROOT = Path("/Users/jun/mocrdown")
DATA_ROOT = Path("/Users/jun/rois-rag/_data/staff_page2")


def load_middle_module() -> None:
    pkg = types.ModuleType("ocrstruct")
    pkg.__path__ = [str(REPO_ROOT / "ocrstruct")]
    sys.modules["ocrstruct"] = pkg

    for name in ["utils", "middle"]:
        path = REPO_ROOT / "ocrstruct" / f"{name}.py"
        spec = importlib.util.spec_from_file_location(f"ocrstruct.{name}", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to load module spec for {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"ocrstruct.{name}"] = module
        spec.loader.exec_module(module)


def main() -> int:
    load_middle_module()

    from ocrstruct.middle import Result

    paths = sorted(DATA_ROOT.glob("**/middle.json"))
    failed: list[tuple[Path, Exception]] = []

    for path in paths:
        try:
            Result.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - debug path
            failed.append((path, exc))

    print(f"files={len(paths)} failed={len(failed)}")
    for path, exc in failed[:10]:
        print("---")
        print(path)
        print(type(exc).__name__)
        print(str(exc)[:4000])

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
