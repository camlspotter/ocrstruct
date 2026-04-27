from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Literal
from typing import NamedTuple
from typing import Any

from mineru.backend.hybrid.hybrid_analyze import doc_analyze as hybrid_doc_analyze
from mineru.backend.pipeline.pipeline_analyze import (
    doc_analyze_streaming as pipeline_doc_analyze_streaming,
)
from mineru.backend.pipeline.pipeline_middle_json_mkcontent import (
    union_make as pipeline_union_make,
)
from mineru.backend.vlm.vlm_analyze import doc_analyze as vlm_doc_analyze
from mineru.backend.vlm.vlm_middle_json_mkcontent import union_make as vlm_union_make
from mineru.cli.common import convert_pdf_bytes_to_bytes, prepare_env
from mineru.data.data_reader_writer import FileBasedDataWriter
from mineru.utils.engine_utils import get_vlm_engine
from mineru.utils.enum_class import MakeMode
from pypdf import PdfReader

from ocrstruct.middle_to_elements import to_elements
from ocrstruct.types import BBox, Element, LinkRegion


logger = logging.getLogger(__name__)


class MineruMarkdownResult(NamedTuple):
    middle_json: dict
    markdown_text: str | None
    extracted_by: str


def middle_json_to_elements(middle_json: dict, *, img_bucket_path: str = "images") -> list[Element]:
    return to_elements(middle_json, img_bucket_path=img_bucket_path)


def elements_to_markdown(
    elements: list[Element],
    *,
    mode: Literal["rag", "html"] = "rag",
) -> str:
    if mode == "rag":
        return "\n".join(element.to_str() for element in elements)
    if mode == "html":
        return "\n".join(element.to_markdown() for element in elements)
    raise ValueError(f"Unsupported markdown mode: {mode}")


def dump_elements_json(elements: list[Element], path: str | Path) -> Path:
    out = Path(path)
    out.write_text(
        json.dumps([element.model_dump(mode="json") for element in elements], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out


def load_elements_json(elements_json_path: str | Path) -> list[Element]:
    path = Path(elements_json_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("elements.json must be a list")
    return [Element.model_validate(item) for item in raw]


def _md_content_to_text(md_content: Any) -> str | None:
    if isinstance(md_content, list):
        return "\n".join(str(x) for x in md_content)
    if md_content is None:
        return None
    return str(md_content)


def render_middle_json_to_markdown(
    middle_json: dict,
    *,
    markdown_image_bucket_path: str = "images",
) -> MineruMarkdownResult:
    pdf_info = middle_json.get("pdf_info")
    if not isinstance(pdf_info, list):
        raise ValueError("middle.json does not have valid 'pdf_info'")

    renderer_attempts: list[tuple[str, Any]] = [
        ("mineru/pipeline", pipeline_union_make),
        ("mineru/vlm", vlm_union_make),
    ]
    for extracted_by, renderer in renderer_attempts:
        try:
            md_content = renderer(
                pdf_info,
                MakeMode.MM_MD,
                markdown_image_bucket_path,
            )
            return MineruMarkdownResult(
                middle_json=middle_json,
                markdown_text=_md_content_to_text(md_content),
                extracted_by=extracted_by,
            )
        except Exception:
            logger.debug("Markdown render via %s failed", extracted_by, exc_info=True)

    logger.warning("Falling back to ocrstruct markdown renderer for middle.json")
    fallback_elements = middle_json_to_elements(
        middle_json,
        img_bucket_path=markdown_image_bucket_path,
    )
    return MineruMarkdownResult(
        middle_json=middle_json,
        markdown_text=elements_to_markdown(fallback_elements),
        extracted_by="ocrstruct/fallback",
    )


def load_middle_json(middle_json_path: str | Path) -> dict:
    path = Path(middle_json_path)
    return json.loads(path.read_text(encoding="utf-8"))


def _pdf_obj_to_str(value: Any, *, max_depth: int = 4) -> str:
    if max_depth <= 0:
        return "..."
    if value is None:
        return "null"
    if isinstance(value, (str, int, float, bool)):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        items = [_pdf_obj_to_str(v, max_depth=max_depth - 1) for v in value[:8]]
        suffix = ", ..." if len(value) > 8 else ""
        return "[" + ", ".join(items) + suffix + "]"
    if isinstance(value, dict):
        keys = list(value.keys())[:8]
        pairs = [
            f"{_pdf_obj_to_str(str(k), max_depth=max_depth - 1)}: "
            f"{_pdf_obj_to_str(value[k], max_depth=max_depth - 1)}"
            for k in keys
        ]
        suffix = ", ..." if len(value) > 8 else ""
        return "{" + ", ".join(pairs) + suffix + "}"
    return json.dumps(str(value), ensure_ascii=False)


def _maybe_bbox_from_rect(rect: Any) -> BBox | None:
    try:
        if rect is None or len(rect) < 4:
            return None
        x0 = float(rect[0])
        y0 = float(rect[1])
        x1 = float(rect[2])
        y1 = float(rect[3])
        return BBox(
            x0=min(x0, x1),
            y0=min(y0, y1),
            x1=max(x0, x1),
            y1=max(y0, y1),
        )
    except Exception:
        return None


def _resolve_dest_page_idx(
    *,
    reader: PdfReader,
    dest_obj: Any,
    page_ref_to_idx: dict[tuple[int, int], int],
) -> int | None:
    if dest_obj is None:
        return None

    # pypdf can resolve Destination-like objects directly in some cases.
    try:
        page_idx = reader.get_destination_page_number(dest_obj)
        if isinstance(page_idx, int) and page_idx >= 0:
            return page_idx
    except Exception:
        pass

    # Named destination string
    try:
        if isinstance(dest_obj, str) and dest_obj in reader.named_destinations:
            named = reader.named_destinations[dest_obj]
            page_idx = reader.get_destination_page_number(named)
            if isinstance(page_idx, int) and page_idx >= 0:
                return page_idx
    except Exception:
        pass

    # Explicit destination array: [page_ref, /XYZ, ...]
    try:
        if isinstance(dest_obj, (list, tuple)) and dest_obj:
            page_ref = dest_obj[0]
            idnum = getattr(page_ref, "idnum", None)
            generation = getattr(page_ref, "generation", None)
            if isinstance(idnum, int) and isinstance(generation, int):
                return page_ref_to_idx.get((idnum, generation))
    except Exception:
        pass

    return None


def extract_pdf_link_regions(pdf_path: str) -> list[LinkRegion]:
    """
    Extract PDF link annotation rectangles.

    Supports:
    - external links: /A /S /URI
    - internal links: /A /S /GoTo and /Dest
    """
    reader = PdfReader(pdf_path)

    page_ref_to_idx: dict[tuple[int, int], int] = {}
    for page_idx, page in enumerate(reader.pages):
        ref = getattr(page, "indirect_reference", None)
        idnum = getattr(ref, "idnum", None)
        generation = getattr(ref, "generation", None)
        if isinstance(idnum, int) and isinstance(generation, int):
            page_ref_to_idx[(idnum, generation)] = page_idx

    out: list[LinkRegion] = []
    for page_idx, page in enumerate(reader.pages):
        annots = page.get("/Annots")
        if annots is None:
            continue
        for annot_ref in annots:
            try:
                annot = annot_ref.get_object()
            except Exception:
                continue
            if str(annot.get("/Subtype")) != "/Link":
                continue

            bbox = _maybe_bbox_from_rect(annot.get("/Rect"))
            if bbox is None:
                continue

            action = annot.get("/A")
            direct_dest = annot.get("/Dest")

            target_kind: str = "unknown"
            uri: str | None = None
            dest_obj: Any = None

            if action is not None:
                action_type = str(action.get("/S"))
                if action_type == "/URI":
                    target_kind = "external"
                    raw_uri = action.get("/URI")
                    if raw_uri is not None:
                        uri = str(raw_uri)
                elif action_type == "/GoTo":
                    target_kind = "internal"
                    dest_obj = action.get("/D")

            if direct_dest is not None:
                target_kind = "internal"
                if dest_obj is None:
                    dest_obj = direct_dest

            dest_page_idx = _resolve_dest_page_idx(
                reader=reader,
                dest_obj=dest_obj,
                page_ref_to_idx=page_ref_to_idx,
            )
            dest_raw = _pdf_obj_to_str(dest_obj) if dest_obj is not None else None

            out.append(
                LinkRegion(
                    page_idx=page_idx,
                    bbox=bbox,
                    target_kind=target_kind,
                    uri=uri,
                    dest_page_idx=dest_page_idx,
                    dest_raw=dest_raw,
                )
            )

    return out


def convert_pdf_to_middle_and_markdown(
    pdf_path: str,
    *,
    tmpdir: str,
    image_dir: str | None = None,
    markdown_image_bucket_path: str = "images",
    backend: str | None = None,
    method: str | None = None,
    lang: str | None = None,
    server_url: str | None = None,
) -> MineruMarkdownResult:
    backend = backend or os.getenv("MINERU_BACKEND", "pipeline")
    method = method or os.getenv("MINERU_METHOD", "auto")
    lang = lang or os.getenv("MINERU_LANG", "japan")
    server_url = server_url or os.getenv("MINERU_SERVER_URL") or None

    with open(pdf_path, "rb") as f:
        file_bytes = f.read()

    pdf_bytes = convert_pdf_bytes_to_bytes(file_bytes, 0, None)
    file_name = Path(pdf_path).stem

    Path(tmpdir).mkdir(parents=True, exist_ok=True)

    if backend == "pipeline":
        if image_dir is None:
            local_image_dir, _local_md_dir = prepare_env(tmpdir, file_name, method)
        else:
            local_image_dir = image_dir
            Path(local_image_dir).mkdir(parents=True, exist_ok=True)
        image_writer = FileBasedDataWriter(local_image_dir)
        middle_json_holder: dict[str, dict] = {}

        def on_doc_ready(
            doc_index: int,
            model_list: list[dict],
            pipeline_middle_json: dict,
            ocr_enable: bool,
        ) -> None:
            del model_list, ocr_enable
            if doc_index == 0:
                middle_json_holder["middle_json"] = pipeline_middle_json

        pipeline_doc_analyze_streaming(
            [pdf_bytes],
            [image_writer],
            [lang],
            on_doc_ready,
            parse_method=method,
            formula_enable=True,
            table_enable=True,
        )
        middle_json = middle_json_holder["middle_json"]
        md_content = pipeline_union_make(
            middle_json["pdf_info"],
            MakeMode.MM_MD,
            markdown_image_bucket_path,
        )
        extracted_by = "mineru/pipeline"

    elif backend.startswith("vlm-"):
        backend_name = backend[4:]
        if backend_name == "auto-engine":
            backend_name = get_vlm_engine(inference_engine="auto", is_async=False)
        if image_dir is None:
            local_image_dir, _local_md_dir = prepare_env(tmpdir, file_name, "vlm")
        else:
            local_image_dir = image_dir
            Path(local_image_dir).mkdir(parents=True, exist_ok=True)
        image_writer = FileBasedDataWriter(local_image_dir)
        middle_json, _infer = vlm_doc_analyze(
            pdf_bytes,
            image_writer=image_writer,
            backend=backend_name,
            server_url=server_url,
        )
        md_content = vlm_union_make(
            middle_json["pdf_info"],
            MakeMode.MM_MD,
            markdown_image_bucket_path,
        )
        extracted_by = f"mineru/vlm:{backend_name}"

    elif backend.startswith("hybrid-"):
        backend_name = backend[7:]
        if backend_name == "auto-engine":
            backend_name = get_vlm_engine(inference_engine="auto", is_async=False)
        parse_method = f"hybrid_{method}"
        if image_dir is None:
            local_image_dir, _local_md_dir = prepare_env(tmpdir, file_name, parse_method)
        else:
            local_image_dir = image_dir
            Path(local_image_dir).mkdir(parents=True, exist_ok=True)
        image_writer = FileBasedDataWriter(local_image_dir)
        middle_json, _infer, _ocr_enabled = hybrid_doc_analyze(
            pdf_bytes,
            image_writer=image_writer,
            backend=backend_name,
            parse_method=parse_method,
            language=lang,
            inline_formula_enable=True,
            server_url=server_url,
        )
        md_content = vlm_union_make(
            middle_json["pdf_info"],
            MakeMode.MM_MD,
            markdown_image_bucket_path,
        )
        extracted_by = f"mineru/hybrid:{backend_name}"

    else:
        raise ValueError("MINERU_BACKEND must be 'pipeline', 'vlm-*', or 'hybrid-*")

    middle_path = Path(tmpdir) / "middle.json"
    middle_path.write_text(
        json.dumps(middle_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"MinerU middle_json saved: {middle_path}")

    return MineruMarkdownResult(
        middle_json=middle_json,
        markdown_text=_md_content_to_text(md_content),
        extracted_by=extracted_by,
    )


def convert_pdf_to_elements(
    pdf_path: str,
    *,
    tmpdir: str,
    img_bucket_path: str = "images",
    backend: str | None = None,
    method: str | None = None,
    lang: str | None = None,
    server_url: str | None = None,
) -> list[Element]:
    result = convert_pdf_to_middle_and_markdown(
        pdf_path,
        tmpdir=tmpdir,
        backend=backend,
        method=method,
        lang=lang,
        server_url=server_url,
    )
    return middle_json_to_elements(result.middle_json, img_bucket_path=img_bucket_path)
