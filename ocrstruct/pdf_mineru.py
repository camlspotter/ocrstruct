from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from mineru.backend.hybrid.hybrid_analyze import doc_analyze as hybrid_doc_analyze
from mineru.backend.pipeline.pipeline_analyze import (
    doc_analyze_streaming as pipeline_doc_analyze_streaming,
)
from mineru.backend.vlm.vlm_analyze import doc_analyze as vlm_doc_analyze
from mineru.cli.common import convert_pdf_bytes_to_bytes
from mineru.data.data_reader_writer import FileBasedDataWriter
from mineru.utils.engine_utils import get_vlm_engine

from ocrstruct.middle import Middle


logger = logging.getLogger(__name__)


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


# def convert_pdf_to_middle_json(
#     pdf_path: str,
#     *,
#     outdir: str,
#     backend: str | None = None,
#     method: str | None = None,
#     lang: str | None = None,
#     server_url: str | None = None,
#     seal_enable: bool = True,
#     formula_enable: bool = True,
#     lazy: bool = False,
# ) -> Result:
#     middle_path = Path(outdir) / "middle.json"
#
#     if lazy and os.path.exists(middle_path):
#         if res := Result.load_json(middle_path):
#             return res
#
#     res = convert_pdf_to_middle(
#         pdf_path,
#         outdir=outdir,
#         backend=backend,
#         method=method,
#         lang=lang,
#         server_url=server_url,
#         seal_enable=seal_enable,
#         formula_enable=formula_enable,
#     )
#
#     res.save_json(middle_path)
#     logger.info(f"MinerU middle_json saved: {middle_path}")
#     return res
#

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
) -> tuple[Middle, str]:
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
        middle_json, extracted_by = _convert_with_pipeline_backend(
            pdf_bytes=pdf_bytes,
            local_image_dir=local_image_dir,
            method=method,
            lang=lang,
            seal_enable=seal_enable,
            formula_enable=formula_enable,
        )
    elif backend.startswith("vlm-"):
        middle_json, extracted_by = _convert_with_vlm_backend(
            pdf_bytes=pdf_bytes,
            local_image_dir=local_image_dir,
            backend_name=backend[4:],
            server_url=server_url,
        )
    elif backend.startswith("hybrid-"):
        middle_json, extracted_by = _convert_with_hybrid_backend(
            pdf_bytes=pdf_bytes,
            local_image_dir=local_image_dir,
            backend_name=backend[7:],
            method=method,
            lang=lang,
            server_url=server_url,
            formula_enable=formula_enable,
        )
    else:
        raise ValueError("MINERU_BACKEND must be 'pipeline', 'vlm-*', or 'hybrid-*'")

    return Middle.model_validate(middle_json), extracted_by


def _convert_with_pipeline_backend(
    *,
    pdf_bytes: bytes,
    local_image_dir: str,
    method: str,
    lang: str,
    seal_enable: bool,
    formula_enable: bool,
) -> tuple[dict[str, Any], str]:
    image_writer = FileBasedDataWriter(local_image_dir)
    middle_json: dict[str, Any] | None = None

    def on_doc_ready(
        doc_index: int,
        model_list: list[dict[str, Any]],
        pipeline_middle_json: dict[str, Any],
        ocr_enable: bool,
    ) -> None:
        nonlocal middle_json
        del model_list, ocr_enable
        if doc_index == 0:
            middle_json = pipeline_middle_json

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

    if middle_json is None:
        raise RuntimeError("Pipeline backend did not produce a middle JSON result")

    return middle_json, "mineru/pipeline"


def _convert_with_vlm_backend(
    *,
    pdf_bytes: bytes,
    local_image_dir: str,
    backend_name: str,
    server_url: str | None,
) -> tuple[dict[str, Any], str]:
    resolved_backend_name = _resolve_vlm_backend_name(backend_name)
    image_writer = FileBasedDataWriter(local_image_dir)
    middle_json, _infer = vlm_doc_analyze(
        pdf_bytes,
        image_writer=image_writer,
        backend=resolved_backend_name,
        server_url=server_url,
    )
    return middle_json, f"mineru/vlm:{resolved_backend_name}"


def _convert_with_hybrid_backend(
    *,
    pdf_bytes: bytes,
    local_image_dir: str,
    backend_name: str,
    method: str,
    lang: str,
    server_url: str | None,
    formula_enable: bool,
) -> tuple[dict[str, Any], str]:
    resolved_backend_name = _resolve_vlm_backend_name(backend_name)
    image_writer = FileBasedDataWriter(local_image_dir)
    middle_json, _infer, _ocr_enabled = hybrid_doc_analyze(
        pdf_bytes,
        image_writer=image_writer,
        backend=resolved_backend_name,
        parse_method=f"hybrid_{method}",
        language=lang,
        inline_formula_enable=formula_enable,
        server_url=server_url,
    )
    return middle_json, f"mineru/hybrid:{resolved_backend_name}"


def _resolve_vlm_backend_name(backend_name: str) -> str:
    if backend_name == "auto-engine":
        return get_vlm_engine(inference_engine="auto", is_async=False)
    return backend_name
