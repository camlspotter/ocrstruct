from __future__ import annotations

import logging
from pathlib import Path

from ocrstruct.html import markdown_to_html, middle_to_html, result_to_html
from ocrstruct.image_understanding import (
    UnderstandingRecord,
    build_images_file,
    compute_middle_json_sha256,
    image_refs_from_middle,
    image_result_from_understanding_record,
    image_understanding_run_from_screening,
    iter_screening_records_from_refs,
    iter_understanding_records_from_screening,
    load_completed_screening_keys,
    load_completed_understanding_keys,
    load_image_refs_from_middle_json,
    load_images_file_json,
    load_pricing_overrides,
    load_screening_records_jsonl,
    load_understanding_records_jsonl,
    merge_understanding_into_middle,
    merge_images_into_middle,
    pricing_for_model,
    screening_record_key,
    understanding_record_key,
)
from ocrstruct.middle import Result, Middle, merge_discarded_blocks
from ocrstruct.middle_to_markdown import RenderOptions, middle_to_markdown, result_to_markdown, render_rag, render_html
from ocrstruct.pdf import convert_pdf_to_middle


logger = logging.getLogger(__name__)


def _merge_image_understanding_if_present(outdir: Path, *, result: Result) -> None:
    understanding_jsonl_path = outdir / "image_understanding.jsonl"
    if not understanding_jsonl_path.exists():
        return

    understanding_records = load_understanding_records_jsonl(understanding_jsonl_path)
    result.middle_json = merge_understanding_into_middle(
        result.middle_json,
        understanding_records,
    )
    logger.info("Merged image understanding: %s", understanding_jsonl_path)


def _write_images_file(
    *,
    outdir: Path,
    middle_json_path: Path,
    understanding_records: list[UnderstandingRecord],
) -> None:
    images_path = outdir / "images.json"
    images_file = build_images_file(
        understanding_records,
        middle_json_path=middle_json_path,
    )
    images_file.save_json(images_path)
    logger.info("Wrote images file: %s", images_path)


def generate_image_understanding(
    *,
    outdir: str | Path,
    pdf_path: str | Path,
    result: Result,
    image_screening_models: list[str],
    image_understanding_models: list[str],
    image_screening_base_url: str | None = None,
    image_screening_api_key: str | None = None,
    image_understanding_base_url: str | None = None,
    image_understanding_api_key: str | None = None,
    model_pricing_json: str | Path | None = None,
    lazy: bool = False,
) -> None:
    if not image_screening_models:
        raise ValueError("image_screening_models must not be empty")
    if not image_understanding_models:
        raise ValueError("image_understanding_models must not be empty")

    resolved_outdir = Path(outdir)
    middle_json_path = resolved_outdir / "middle.json"
    refs = image_refs_from_middle(
        result.middle_json,
        pdf_path=str(pdf_path),
        middle_json_path=str(middle_json_path),
    )
    if not refs:
        logger.info("No image refs found for image understanding")
        return

    pricing_overrides = load_pricing_overrides(model_pricing_json)
    screening_path = resolved_outdir / "image_screening.jsonl"
    screening_path.parent.mkdir(parents=True, exist_ok=True)
    screening_mode = "a" if lazy and screening_path.exists() else "w"
    completed_screening_keys = (
        load_completed_screening_keys(screening_path)
        if lazy and screening_path.exists()
        else set()
    )
    with screening_path.open(screening_mode, encoding="utf-8") as handle:
        for model in image_screening_models:
            pricing = pricing_for_model(model, pricing_overrides)
            for record in iter_screening_records_from_refs(
                refs,
                model=model,
                pricing=pricing,
                base_url=image_screening_base_url,
                api_key=image_screening_api_key,
                thinking=False,
                existing_keys=completed_screening_keys,
            ):
                handle.write(record.model_dump_json() + "\n")
                handle.flush()
                if record.status.ok:
                    completed_screening_keys.add(screening_record_key(record))
    logger.info("Updated image screening results: %s", screening_path)

    screening_records = load_screening_records_jsonl(screening_path)
    if not screening_records:
        logger.info("No successful screening records found for image understanding")
        return

    understanding_path = resolved_outdir / "image_understanding.jsonl"
    understanding_mode = "a" if lazy and understanding_path.exists() else "w"
    completed_understanding_keys = (
        load_completed_understanding_keys(understanding_path)
        if lazy and understanding_path.exists()
        else set()
    )
    with understanding_path.open(understanding_mode, encoding="utf-8") as handle:
        for model in image_understanding_models:
            pricing = pricing_for_model(model, pricing_overrides)
            for record in iter_understanding_records_from_screening(
                screening_records,
                model=model,
                pricing=pricing,
                base_url=image_understanding_base_url,
                api_key=image_understanding_api_key,
                thinking=False,
                existing_keys=completed_understanding_keys,
            ):
                handle.write(record.model_dump_json() + "\n")
                handle.flush()
                if record.status.ok:
                    completed_understanding_keys.add(understanding_record_key(record))
    logger.info("Updated image understanding results: %s", understanding_path)
    understanding_records = load_understanding_records_jsonl(understanding_path)
    _write_images_file(
        outdir=resolved_outdir,
        middle_json_path=middle_json_path,
        understanding_records=understanding_records,
    )


def convert_one_pdf(
    *,
    pdf_path: str | Path,
    outdir: str | Path | None = None,
    backend: str | None = None,
    method: str | None = None,
    lang: str | None = None,
    server_url: str | None = None,
    seal_enable: bool = True,
    formula_enable: bool = True,
    lazy: bool = False,
    with_image_understanding: bool = False,
    image_screening_models: list[str] | None = None,
    image_screening_base_url: str | None = None,
    image_screening_api_key: str | None = None,
    image_understanding_models: list[str] | None = None,
    image_understanding_base_url: str | None = None,
    image_understanding_api_key: str | None = None,
    model_pricing_json: str | Path | None = None,
) -> Result:
    resolved_pdf_path = Path(pdf_path)
    if not resolved_pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {resolved_pdf_path}")

    resolved_outdir = Path(outdir) if outdir is not None else resolved_pdf_path.with_suffix("")
    resolved_outdir.mkdir(parents=True, exist_ok=True)
    (resolved_outdir / "images").mkdir(parents=True, exist_ok=True)

    result = convert_pdf_to_middle(
        str(resolved_pdf_path),
        outdir=str(resolved_outdir),
        backend=backend,
        method=method,
        lang=lang,
        server_url=server_url,
        seal_enable=seal_enable,
        formula_enable=formula_enable,
        lazy=lazy,
    )

    result.middle_json = merge_discarded_blocks(result.middle_json)

    if with_image_understanding:
        if not image_screening_models:
            raise ValueError("with_image_understanding requires image_screening_models")
        if not image_understanding_models:
            raise ValueError("with_image_understanding requires image_understanding_models")
        generate_image_understanding(
            outdir=resolved_outdir,
            pdf_path=resolved_pdf_path,
            result=result,
            image_screening_models=image_screening_models,
            image_understanding_models=image_understanding_models,
            image_screening_base_url=image_screening_base_url,
            image_screening_api_key=image_screening_api_key,
            image_understanding_base_url=image_understanding_base_url,
            image_understanding_api_key=image_understanding_api_key,
            model_pricing_json=model_pricing_json,
            lazy=lazy,
        )
    _merge_image_understanding_if_present(resolved_outdir, result=result)
    return result


def render_result(outdir : Path, result : Result):
    markdown_text = result_to_markdown(
        result,
        options= render_rag,
    )
    text_md = outdir / "text.md"
    text_md.write_text(markdown_text, encoding="utf-8")
    logger.info("wrote %s", text_md)

    html_text = result_to_html(
        result, 
        options= render_html,
    )
    if html_text is None:
        logger.info("pandoc not found or failed; skip HTML conversion")
    else:
        text_html = outdir / "text.html"
        text_html.write_text(html_text, encoding="utf-8")
        logger.info("wrote %s", text_html)


__all__ = [
    "build_images_file",
    "compute_middle_json_sha256",
    "convert_one_pdf",
    "convert_pdf_to_middle",
    "generate_image_understanding",
    "image_refs_from_middle",
    "image_result_from_understanding_record",
    "image_understanding_run_from_screening",
    "iter_screening_records_from_refs",
    "iter_understanding_records_from_screening",
    "load_completed_screening_keys",
    "load_completed_understanding_keys",
    "load_image_refs_from_middle_json",
    "load_images_file_json",
    "load_pricing_overrides",
    "load_screening_records_jsonl",
    "load_understanding_records_jsonl",
    "markdown_to_html",
    "merge_discarded_blocks",
    "merge_images_into_middle",
    "merge_understanding_into_middle",
    "middle_to_html",
    "middle_to_markdown",
    "pricing_for_model",
    "result_to_html",
    "result_to_markdown",
]
