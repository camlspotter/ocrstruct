from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import NamedTuple

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

from mocrdown.middle_to_elements import to_elements
from mocrdown.types import Element


logger = logging.getLogger(__name__)


class MineruMarkdownResult(NamedTuple):
    middle_json: dict
    markdown_text: str | None
    extracted_by: str


def middle_json_to_elements(middle_json: dict, *, img_bucket_path: str = "images") -> list[Element]:
    return to_elements(middle_json, img_bucket_path=img_bucket_path)


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

    markdown_text: str | None
    if isinstance(md_content, list):
        markdown_text = "\n".join(str(x) for x in md_content)
    elif md_content is None:
        markdown_text = None
    else:
        markdown_text = str(md_content)

    return MineruMarkdownResult(
        middle_json=middle_json,
        markdown_text=markdown_text,
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
