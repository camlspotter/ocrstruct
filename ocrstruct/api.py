from __future__ import annotations

import logging
from pathlib import Path

from ocrstruct.middle_to_html import middle_to_html
from ocrstruct.image_understanding import (
    UnderstandingRecord,
    build_images_file,
    image_refs_from_middle,
    iter_screening_records_from_refs,
    iter_understanding_records_from_screening,
    load_completed_screening_keys,
    load_completed_understanding_keys,
    load_pricing_overrides,
    load_screening_records_jsonl,
    load_understanding_records_jsonl,
    merge_understanding_into_middle,
    pricing_for_model,
    screening_record_key,
    understanding_record_key,
)
from ocrstruct.middle import Middle, merge_discarded_blocks
from ocrstruct.middle_to_markdown import RenderOptions, middle_to_markdown, render_rag, render_html
from ocrstruct.pdf_mineru import convert_pdf_to_middle
from ocrstruct.result import Result, Parameters
from ocrstruct.utils import BaseModelWithSave, sha256_file


logger = logging.getLogger(__name__)


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
    middle: Middle,
    image_screening_model: str,
    image_screening_base_url: str | None = None,
    image_screening_api_key: str | None = None,
    image_understanding_model: str,
    image_understanding_base_url: str | None = None,
    image_understanding_api_key: str | None = None,
    model_pricing_json: str | Path | None = None,
    lazy: bool = False,
) -> list[UnderstandingRecord]:
    resolved_outdir = Path(outdir)
    middle_json_path = resolved_outdir / "middle.json"
    refs = image_refs_from_middle(
        middle,
        pdf_path=str(pdf_path),
        middle_json_path=str(middle_json_path),
    )
    if not refs:
        logger.info("No image refs found for image understanding")
        return []

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
        pricing = pricing_for_model(image_screening_model, pricing_overrides)
        for record in iter_screening_records_from_refs(
            refs,
            model=image_screening_model,
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
        return []

    understanding_path = resolved_outdir / "image_understanding.jsonl"
    understanding_mode = "a" if lazy and understanding_path.exists() else "w"
    completed_understanding_keys = (
        load_completed_understanding_keys(understanding_path)
        if lazy and understanding_path.exists()
        else set()
    )
    with understanding_path.open(understanding_mode, encoding="utf-8") as handle:
        pricing = pricing_for_model(image_understanding_model, pricing_overrides)
        for record in iter_understanding_records_from_screening(
            screening_records,
            model=image_understanding_model,
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
    return understanding_records


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
    with_image_understanding: bool = False,
    image_screening_model: str | None = None,
    image_screening_base_url: str | None = None,
    image_screening_api_key: str | None = None,
    image_understanding_model: str | None = None,
    image_understanding_base_url: str | None = None,
    image_understanding_api_key: str | None = None,
    model_pricing_json: str | Path | None = None,
) -> Result:
    parameters = Parameters(
        source_checksum= sha256_file(pdf_path),
        backend= backend,
        method= method,
        lang= lang,
        seal_enable= seal_enable,
        formula_enable= formula_enable,
        with_image_understanding= with_image_understanding,
        image_screening_model= image_screening_model,
        image_understanding_model= image_understanding_model,
    )

    resolved_pdf_path = Path(pdf_path)
    if not resolved_pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {resolved_pdf_path}")

    resolved_outdir = Path(outdir) if outdir is not None else resolved_pdf_path.with_suffix("")
    resolved_outdir.mkdir(parents=True, exist_ok=True)
    (resolved_outdir / "images").mkdir(parents=True, exist_ok=True)

    middle_json_path = resolved_outdir / 'middle.json'

    # Do it lazy
    res : Result | None = None
    try:
        res = Result.load_json(middle_json_path)
        if res.source_path == str(resolved_pdf_path) and res.parameters == parameters:
            return res
    except:
        pass

    # Extract texts lazily
    if (
        res
        and res.source_path == str(resolved_pdf_path)
        and res.parameters.without_image_understanding() == parameters.without_image_understanding()
    ):
        middle= res.middle
        extracted_by= res.extracted_by
    else:
        middle, extracted_by = convert_pdf_to_middle(
            str(resolved_pdf_path),
            outdir=str(resolved_outdir),
            backend=backend,
            method=method,
            lang=lang,
            server_url=server_url,
            seal_enable=seal_enable,
            formula_enable=formula_enable,
        )
        middle = merge_discarded_blocks(middle)

    if with_image_understanding:
        if not image_screening_model:
            raise ValueError("with_image_understanding requires image_screening_model")
        if not image_understanding_model:
            raise ValueError("with_image_understanding requires image_understanding_model")
        understandings = generate_image_understanding(
            outdir=resolved_outdir,
            pdf_path=resolved_pdf_path,
            middle= middle,
            image_screening_model=image_screening_model,
            image_screening_base_url=image_screening_base_url,
            image_screening_api_key=image_screening_api_key,
            image_understanding_model=image_understanding_model,
            image_understanding_base_url=image_understanding_base_url,
            image_understanding_api_key=image_understanding_api_key,
            model_pricing_json=model_pricing_json,
        )
        middle = merge_understanding_into_middle(middle, understandings)

    return Result(
        middle= middle,
        source_path= str(pdf_path),
        extracted_by= extracted_by,
        parameters= parameters,
    )


def render_middle(outdir : Path, middle: Middle):
    markdown_text = middle_to_markdown(middle, options= render_rag)
    text_md = outdir / "text.md"
    text_md.write_text(markdown_text, encoding="utf-8")
    logger.info("wrote %s", text_md)

    html_text = middle_to_html(middle, options= render_html)
    if html_text is None:
        logger.info("pandoc not found or failed; skip HTML conversion")
    else:
        text_html = outdir / "text.html"
        text_html.write_text(html_text, encoding="utf-8")
        logger.info("wrote %s", text_html)
