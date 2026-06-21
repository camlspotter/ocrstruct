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

from ocrstruct.middle import Middle, Result


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
) -> Result:
    middle_path = Path(outdir) / "middle.json"

    if lazy and os.path.exists(middle_path):
        if res := Result.load_json(middle_path):
            return res

    res = _convert_pdf_to_middle_impl(
        pdf_path,
        outdir=outdir,
        backend=backend,
        method=method,
        lang=lang,
        server_url=server_url,
        seal_enable=seal_enable,
        formula_enable=formula_enable,
    )

    res.save_json(middle_path)
    logger.info(f"MinerU middle_json saved: {middle_path}")
    return res


def convert_pdf_to_middle_json(
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
) -> Result:
    return convert_pdf_to_middle(
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

    return Result(
        middle_json=Middle.model_validate(middle_json),
        extracted_by=extracted_by,
    )
