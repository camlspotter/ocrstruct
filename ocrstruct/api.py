from __future__ import annotations

import logging
from pathlib import Path

from ocrstruct.middle_to_html import middle_to_html
from ocrstruct.image_understanding import analyze_images_and_embed_into_middle
from ocrstruct.vlm import VLMConfig, pricing_for_model
from ocrstruct.middle import Middle, merge_discarded_blocks
from ocrstruct.middle_to_markdown import RenderOptions, middle_to_markdown, render_rag, render_html
from ocrstruct.pdf_mineru import convert_pdf_to_middle
from ocrstruct.result import Result, Parameters
from ocrstruct.utils import BaseModelWithSave, sha256_file


logger = logging.getLogger(__name__)


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

    if with_image_understanding:
        if not image_screening_model:
            raise ValueError("with_image_understanding requires image_screening_model")
        if not image_understanding_model:
            raise ValueError("with_image_understanding requires image_understanding_model")

        screening_vlm_config = VLMConfig(
            model= image_screening_model,
            thinking= False,
            base_url= image_screening_base_url,
            api_key= image_screening_api_key,
            pricing= pricing_for_model(image_screening_model),
        )
        understanding_vlm_config = VLMConfig(
            model= image_understanding_model,
            thinking= False,
            base_url= image_understanding_base_url,
            api_key= image_understanding_api_key,
            pricing= pricing_for_model(image_understanding_model),
        )
        middle = merge_discarded_blocks(middle)
        middle = analyze_images_and_embed_into_middle(
            middle, 
            str(resolved_pdf_path),
            outdir=str(resolved_outdir),
            screening_vlm_config= screening_vlm_config,
            understanding_vlm_config= understanding_vlm_config,
        )

    result = Result(
        middle= middle,
        source_path= str(pdf_path),
        extracted_by= extracted_by,
        parameters= parameters,
    )
    result.save_json(middle_json_path)
    return result


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
