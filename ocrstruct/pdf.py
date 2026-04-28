from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Literal
from typing import NamedTuple
from typing import Any
from pydantic import BaseModel

from mineru.backend.hybrid.hybrid_analyze import doc_analyze as hybrid_doc_analyze
from mineru.backend.pipeline.pipeline_analyze import (
    doc_analyze_streaming as pipeline_doc_analyze_streaming,
)
from mineru.backend.pipeline.pipeline_middle_json_mkcontent import (
    union_make as pipeline_union_make,
)
from mineru.backend.vlm.vlm_analyze import doc_analyze as vlm_doc_analyze
from mineru.backend.vlm.vlm_middle_json_mkcontent import union_make as vlm_union_make
from mineru.cli.common import convert_pdf_bytes_to_bytes
from mineru.data.data_reader_writer import FileBasedDataWriter
from mineru.utils.engine_utils import get_vlm_engine
from mineru.utils.enum_class import MakeMode
from pypdf import PdfReader

from ocrstruct.middle_to_elements import middle_to_elements
from ocrstruct.types import BBox, Element, LinkRegion
from ocrstruct.utils import BaseModelWithSave, load_json, save_json


logger = logging.getLogger(__name__)


class Result(BaseModelWithSave):
    middle_json: dict
    extracted_by: str


class _NoopSealOcrModel:
    def ocr(self, *args: Any, **kwargs: Any) -> list[list]:
        del args, kwargs
        return [[]]


@contextmanager
def _maybe_disable_pipeline_seal_ocr(disabled: bool):
    if not disabled:
        yield
        return

    from mineru.backend.pipeline.model_init import AtomModelSingleton
    from mineru.backend.pipeline.model_list import AtomicModel

    original_get_atom_model = AtomModelSingleton.get_atom_model
    noop_model = _NoopSealOcrModel()

    def patched_get_atom_model(self, atom_model_name: str, **kwargs: Any):
        if atom_model_name == AtomicModel.OCR and kwargs.get("lang") == "seal":
            logger.info("Skipping MinerU seal OCR because seal_enable=False")
            return noop_model
        return original_get_atom_model(self, atom_model_name, **kwargs)

    AtomModelSingleton.get_atom_model = patched_get_atom_model
    try:
        yield
    finally:
        AtomModelSingleton.get_atom_model = original_get_atom_model


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


def convert_pdf_to_middle(
    pdf_path: str,
    *,
    outdir: str,
    backend: str | None = None,
    method: str | None = None,
    lang: str | None = None,
    server_url: str | None = None,
    seal_enable: bool = True,
    formula_enable: bool = True,
    lazy: bool = False,
    fork: bool | None = None,
) -> Result:
    middle_path = Path(outdir) / "middle.json"

    if lazy and os.path.exists(middle_path):
        if res := Result.load_json(middle_path):
            return res

    if fork is None:
        fork = os.getenv("OCRSTRUCT_FORK_PDF_TO_MIDDLE", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    if fork:
        return _convert_pdf_to_middle_forked(
            pdf_path,
            outdir=outdir,
            backend=backend,
            method=method,
            lang=lang,
            server_url=server_url,
            seal_enable=seal_enable,
            formula_enable=formula_enable,
            lazy=lazy,
        )

    return _convert_pdf_to_middle_impl(
        pdf_path,
        outdir=outdir,
        backend=backend,
        method=method,
        lang=lang,
        server_url=server_url,
        seal_enable=seal_enable,
        formula_enable=formula_enable,
    )

def _convert_pdf_to_middle_impl(
    pdf_path: str,
    *,
    outdir: str,
    backend: str | None = None,
    method: str | None = None,
    lang: str | None = None,
    server_url: str | None = None,
    seal_enable: bool = True,
    formula_enable: bool = True,
) -> Result:
    middle_path = Path(outdir) / "middle.json"
    backend = backend or os.getenv("MINERU_BACKEND", "pipeline")
    method = method or os.getenv("MINERU_METHOD", "auto")
    lang = lang or os.getenv("MINERU_LANG", "japan")
    server_url = server_url or os.getenv("MINERU_SERVER_URL") or None

    with open(pdf_path, "rb") as f:
        file_bytes = f.read()

    pdf_bytes = convert_pdf_bytes_to_bytes(file_bytes, 0, None)
    Path(outdir).mkdir(parents=True, exist_ok=True)
    local_image_dir = str(Path(outdir) / "images")
    Path(local_image_dir).mkdir(parents=True, exist_ok=True)

    if backend == "pipeline":
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

        with _maybe_disable_pipeline_seal_ocr(disabled=not seal_enable):
            pipeline_doc_analyze_streaming(
                [pdf_bytes],
                [image_writer],
                [lang],
                on_doc_ready,
                parse_method=method,
                formula_enable=formula_enable,
                table_enable=True,
            )
        middle_json = middle_json_holder["middle_json"]
        extracted_by = "mineru/pipeline"

    elif backend.startswith("vlm-"):
        backend_name = backend[4:]
        if backend_name == "auto-engine":
            backend_name = get_vlm_engine(inference_engine="auto", is_async=False)
        image_writer = FileBasedDataWriter(local_image_dir)
        middle_json, _infer = vlm_doc_analyze(
            pdf_bytes,
            image_writer=image_writer,
            backend=backend_name,
            server_url=server_url,
        )
        extracted_by = f"mineru/vlm:{backend_name}"

    elif backend.startswith("hybrid-"):
        backend_name = backend[7:]
        if backend_name == "auto-engine":
            backend_name = get_vlm_engine(inference_engine="auto", is_async=False)
        parse_method = f"hybrid_{method}"
        image_writer = FileBasedDataWriter(local_image_dir)
        middle_json, _infer, _ocr_enabled = hybrid_doc_analyze(
            pdf_bytes,
            image_writer=image_writer,
            backend=backend_name,
            parse_method=parse_method,
            language=lang,
            inline_formula_enable=formula_enable,
            server_url=server_url,
        )
        extracted_by = f"mineru/hybrid:{backend_name}"
    else:
        raise ValueError("MINERU_BACKEND must be 'pipeline', 'vlm-*', or 'hybrid-*")

    res = Result(
        middle_json=middle_json,
        extracted_by=extracted_by,
    )

    res.save_json(middle_path)
    logger.info(f"MinerU middle_json saved: {middle_path}")
    return res


def _convert_pdf_to_middle_child(
    error_queue: Any,
    pdf_path: str,
    *,
    outdir: str,
    backend: str | None,
    method: str | None,
    lang: str | None,
    server_url: str | None,
    seal_enable: bool,
    formula_enable: bool,
) -> None:
    try:
        _convert_pdf_to_middle_impl(
            pdf_path,
            outdir=outdir,
            backend=backend,
            method=method,
            lang=lang,
            server_url=server_url,
            seal_enable=seal_enable,
            formula_enable=formula_enable,
        )
    except BaseException:
        error_queue.put(traceback.format_exc())
        raise


def _convert_pdf_to_middle_forked(
    pdf_path: str,
    *,
    outdir: str,
    backend: str | None,
    method: str | None,
    lang: str | None,
    server_url: str | None,
    seal_enable: bool,
    formula_enable: bool,
    lazy: bool,
) -> Result:
    middle_path = Path(outdir) / "middle.json"
    if lazy and os.path.exists(middle_path):
        if res := Result.load_json(middle_path):
            return res

    ctx = mp.get_context("spawn")
    error_queue = ctx.Queue()
    proc = ctx.Process(
        target=_convert_pdf_to_middle_child,
        args=(error_queue, pdf_path),
        kwargs={
            "outdir": outdir,
            "backend": backend,
            "method": method,
            "lang": lang,
            "server_url": server_url,
            "seal_enable": seal_enable,
            "formula_enable": formula_enable,
        },
    )
    proc.start()
    proc.join()

    error_text: str | None = None
    if not error_queue.empty():
        error_text = error_queue.get()

    if proc.exitcode != 0:
        if error_text is not None:
            raise RuntimeError(
                f"forked convert_pdf_to_middle failed for {pdf_path}\n{error_text}"
            )
        raise RuntimeError(
            f"forked convert_pdf_to_middle failed for {pdf_path} with exit code {proc.exitcode}"
        )

    return Result.load_json(middle_path)

def convert_pdf_to_elements(
    pdf_path: str,
    *,
    outdir: str,
    backend: str | None = None,
    method: str | None = None,
    lang: str | None = None,
    server_url: str | None = None,
    seal_enable: bool = True,
    formula_enable: bool = True,
    lazy: bool = False,
    fork: bool | None = None,
) -> list[Element]:
    elements_json_path = Path(outdir) / 'elements.json'
    elements : list[Element] | None = None
    if lazy and os.path.exists(elements_json_path):
        if elements := load_json(list[Element], elements_json_path):
            return elements
    result = convert_pdf_to_middle(pdf_path, outdir= outdir, backend= backend, method= method, lang= lang, server_url= server_url, seal_enable= seal_enable, formula_enable=formula_enable, lazy= lazy, fork=fork)
    elements = middle_to_elements(result.middle_json)
    assert elements
    save_json(list[Element], elements_json_path, elements)
    return elements
