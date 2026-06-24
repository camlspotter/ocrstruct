from __future__ import annotations

import base64
import json
import logging
import mimetypes
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Literal, TypeAlias, cast

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError, TypeAdapter

from ocrstruct.middle import Block, ImageUnderstandingSummary, Middle, Span, normalize_text
from ocrstruct.utils import BaseModelWithSave, sha256_file
from ocrstruct.result import Parameters
from ocrstruct.image_screening import (
    ImageRef, ImageKind, RagValue, DetailLevel, ScreeningResult, 
    screening_context_text,
    image_refs_from_middle,
    run_image_screening,
)
from ocrstruct.vlm import (
    TokenUsage, PriceEstimate, ModelPricing, VLM, VLMConfig, 
    DEFAULT_MODEL_PRICING,
    image_json_request, 
    estimate_price_from_completion
)


logger = logging.getLogger(__name__)


class UnderstandingResult(BaseModel):
    keywords: list[str]
    short_description: str|None = None
    long_description: str|None = None
    notes: str | None = None


class UnderstandingRunResult(BaseModel):
    vlm: VLM
    started_at: str
    raw_text: str
    result: UnderstandingResult
    usage: TokenUsage | None = None
    price: PriceEstimate | None = None


Cache = dict[tuple[VLMConfig, VLMConfig, ImageRef], tuple[ScreeningResult, UnderstandingResult]]
CacheLine = TypeAdapter(tuple[VLMConfig, VLMConfig, ImageRef, ScreeningResult, UnderstandingResult])


def cache_path(outdir : str) -> str:
    return outdir + '/' + 'image_understanding_cache.jsonl'


def load_cache(outdir : str) -> Cache:
    try:
        cache : Cache = {}
        path = cache_path(outdir)
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            (screening_vlm_config, understanding_vlm_config, ref, screening_result, understanding_result) = CacheLine.validate_json(line)
            cache[(screening_vlm_config, understanding_vlm_config, ref)] = (screening_result, understanding_result)
        return cache
    except:
        return {}


def append_to_cache(
    handle, 
    screening_vlm_config:VLMConfig, 
    understanding_vlm_config: VLMConfig,
    ref: ImageRef,
    screening_result: ScreeningResult,
    understanding_result: UnderstandingResult,
) -> None:
    handle.write(CacheLine.dump_json(
        (screening_vlm_config, understanding_vlm_config, ref, screening_result, understanding_result), 
        ensure_ascii=False
    ).decode('utf-8'))
    handle.write('\n')


def _understanding_context_text(ref: ImageRef, screening: ScreeningResult) -> str:
    parts = [
        screening_context_text(ref),
        "",
        "Screening result:",
        f"- kind: {screening.kind}",
        f"- rag_value: {screening.rag_value}",
        f"- detail_level: {screening.detail_level}",
    ]
    if screening.notes is not None:
        parts.append(f"- screening_notes: {screening.notes}")
    return "\n".join(parts)


def _kind_specific_understanding_guidance(
    kind: ImageKind,
    detail_level: DetailLevel,
) -> str:
    match kind:
        case "diagram":
            lines = [
                "- This image is a diagram, flowchart, organization chart, or schematic.",
                "- In long_description, explain the main structure so that a reader can understand the overall flow or relationships without seeing the image.",
                "- For process diagrams or flowcharts, explain the main steps in order and the major branch conditions when visible.",
                "- For organization charts, preserve important names, department names, and role names when they are readable.",
                "- Do not try to fully transcribe every box or every small label into long_description.",
            ]
        case "chart_or_graph":
            lines = [
                "- This image is a chart or graph.",
                "- In long_description, include the chart topic, axis labels, units, legend or series names, and the most important comparisons or trends when visible.",
                "- Prefer meaning and structure over raw OCR-like text.",
                "- If numeric comparisons are important and clearly readable, summarize the important values or rankings in long_description.",
                "- If the chart can be translated into a small markdown table without guessing, prefer that interpretation mentally, but still return plain JSON fields here.",
            ]
        case "table_or_form":
            lines = [
                "- This image is a table or form.",
                "- In long_description, explain what the table or form is for, its main columns, rows, or fields, and the kinds of information it contains.",
                "- Preserve important headers, field names, and key values when readable.",
            ]
        case "ui_or_screenshot":
            lines = [
                "- This image is a UI screenshot or screen-oriented diagram.",
                "- In long_description, explain the purpose of the screen, the main UI elements, and the primary user actions or workflow shown.",
            ]
        case "text_as_image":
            lines = [
                "- This image is mainly text rendered as an image.",
                "- In long_description, explain the document type, topic, and the most important points.",
            ]
        case "code_symbol":
            lines = [
                "- This image is a QR code, barcode, or similar code symbol.",
                "- Do not guess the decoded payload from visual appearance alone.",
                "- In long_description, simply state that the image contains a machine-readable code and mention any visible surrounding text or context.",
            ]
        case "seal" | "logo_or_mark" | "decorative" | "arrow_only":
            lines = [
                "- This image is low-detail or decorative.",
                "- Keep long_description compact and focused on the visible role or appearance of the image.",
                "- Avoid over-interpreting minor visual details.",
            ]
        case _:
            lines = [
                "- Focus on the information most useful for later retrieval and human understanding.",
                "- Preserve visible labels or text only when they add clear value.",
            ]

    return '\n'.join(lines)


def _image_understanding_prompt(kind: ImageKind, detail_level: DetailLevel) -> str:
    kind_guidance = _kind_specific_understanding_guidance(kind, detail_level)
    if detail_level == "short":
        return f"""
You analyze a document image after screening for RAG enrichment.

Return strict JSON with this shape:
{{
  "keywords": [string, ...],
  "short_description": string,
  "notes": string | null
}}

Rules:
- Write short_description in Japanese.
- short_description should be concise and no more than 120 Japanese characters.
- Summarize only the main content and purpose of the image.
- Do not include exhaustive detail.
- Return keywords as a JSON array of short Japanese search terms or short phrases.
- Keywords should help BM25/RAG retrieval and should focus on distinctive nouns, labels, UI terms, entity names, titles, or topic words.
- Include all important searchable terms that seem useful for later retrieval.
- Do not include generic filler words. Avoid duplicates.
- notes are only for caveats or limitations in the result.
- Use notes for issues such as unreadable text, cropped content, or uncertainty.
- If there is no important caveat, return null.
{kind_guidance}
""".strip()
    if detail_level == "long":
        return f"""
You analyze a document image after screening for RAG enrichment.

Return strict JSON with this shape:
{{
  "keywords": [string, ...],
  "short_description": string,
  "long_description": string,
  "notes": string | null
}}

Rules:
- Write all fields in Japanese except raw copied text is not needed here.
- short_description should be concise and no more than 120 Japanese characters.
- long_description should be detailed enough that a reader can understand the main content without seeing the image, and should usually stay within 400 Japanese characters.
- short_description should summarize the image in one compact description.
- long_description should explain the important structure, relations, labels, and meaning.
- Do not try to fully transcribe every visible word into long_description.
- Return keywords as a JSON array of short Japanese search terms or short phrases.
- Keywords should help BM25/RAG retrieval and should preserve the most important labels, titles, entity names, topic words, UI terms, step names, axis labels, legend names, or department names when relevant.
- Include all important searchable terms that seem useful for later retrieval.
- Do not include generic filler words. Avoid duplicates.
- notes are only for caveats or limitations in the result.
- Use notes for issues such as unreadable text, cropped content, or uncertainty.
- If there is no important caveat, return null.
{kind_guidance}
""".strip()
    return f"""
You analyze a document image after screening for RAG enrichment.

Return strict JSON with this shape:
{{
  "keywords": [string, ...],
  "short_description": string,
  "long_description": string,
  "notes": string | null
}}

Rules:
- Write short_description and long_description in Japanese.
- short_description should be concise and no more than 120 Japanese characters.
- long_description should be detailed enough that a reader can understand the main content without seeing the image, and should usually stay within 400 Japanese characters.
- Return keywords as a JSON array of short Japanese search terms or short phrases.
- Keywords should help BM25/RAG retrieval and should preserve the most important labels, titles, entity names, topic words, UI terms, step names, axis labels, legend names, department names, or other searchable terms when relevant.
- Include all important searchable terms that seem useful for later retrieval.
- Do not include generic filler words. Avoid duplicates.
- notes are only for caveats or limitations in the result.
- Use notes for issues such as unreadable text, cropped content, or uncertainty.
- If there is no important caveat, return null.
{kind_guidance}
""".strip()


def run_image_understanding(
    vlm_config: VLMConfig,
    ref: ImageRef,
    screening: ScreeningResult,
) -> UnderstandingRunResult:
    started_at = datetime.now(UTC).isoformat()
    if screening.detail_level == "skip":
        return UnderstandingRunResult(
            vlm= vlm_config.vlm,
            started_at= started_at,
            raw_text= 'n/a',
            result= UnderstandingResult(
               keywords= [],
                notes= screening.notes,
                short_description= None,
                long_description= None,
            ),
            usage= TokenUsage(input_tokens=0, output_tokens=0, total_tokens=0),
            price= PriceEstimate(
                input_cost_usd= 0,
                output_cost_usd= 0,
                total_cost_usd= 0,
            )
        )

    client = OpenAI(api_key=vlm_config.api_key, base_url=vlm_config.base_url,)
    prompt = _image_understanding_prompt(screening.kind, screening.detail_level)
    request = image_json_request(
        vlm= vlm_config.vlm,
        prompt=prompt,
        context_text=_understanding_context_text(ref, screening),
        image_data_url= ref.image_data_url,
        schema_name=f"image_understanding_{screening.detail_level}",
        schema_model= UnderstandingResult,
    )
    last_error: Exception | None = None
    for _attempt in range(1):
        completion = client.chat.completions.create(**cast(Any, request))
        content = completion.choices[0].message.content
        if content is None:
            last_error = ValueError("Model returned no content")
            continue
        try:
            result = UnderstandingResult.model_validate_json(content)
            usage, price = estimate_price_from_completion(completion, vlm_config.pricing)
            return UnderstandingRunResult(
                vlm= vlm_config.vlm,
                started_at=started_at,
                raw_text=content,
                result=result,
                usage=usage,
                price=price,
            )
        except (ValidationError, ValueError, json.JSONDecodeError) as error:
            last_error = error
            continue
    if last_error is None:
        raise ValueError("Model returned no content")
    raise last_error


def analyze_images_and_embed_into_middle(
    middle: Middle, 
    pdf_path: str, 
    outdir: str,
    screening_vlm_config: VLMConfig,
    understanding_vlm_config: VLMConfig,
) -> Middle:
    dict = _analyze_images(middle, pdf_path, outdir, screening_vlm_config, understanding_vlm_config)
    return _embed_into_middle(middle, dict)


def _analyze_images(
    middle: Middle, 
    pdf_path: str, 
    outdir: str,
    screening_vlm_config: VLMConfig,
    understanding_vlm_config: VLMConfig,
) -> dict[ImageRef, tuple[ScreeningResult, UnderstandingResult]]:
    cache = load_cache(outdir)
    results = {}
    with open(cache_path(outdir), 'a') as f:
        for ref in image_refs_from_middle(middle, pdf_path= pdf_path, outdir= outdir):
            cache_key = (screening_vlm_config, understanding_vlm_config, ref)

            result : tuple[ScreeningResult, UnderstandingResult] | None
            if result := cache.get(cache_key, None):
                pass
            else:
                screening_result = run_image_screening(screening_vlm_config, ref).result
                understanding_result = run_image_understanding(understanding_vlm_config, ref, screening_result).result
                result = (screening_result, understanding_result)
                cache[cache_key] = result
                append_to_cache(f, screening_vlm_config, understanding_vlm_config, ref, screening_result, understanding_result)
            screening_result, understanding_result = result
            results[ref] = (screening_result, understanding_result)
    return results


def _embed_into_middle(
    middle: Middle, 
    results: dict[ImageRef, tuple[ScreeningResult, UnderstandingResult]]
) -> Middle:
    dict : dict[tuple[int, int | None, str], ImageUnderstandingSummary] = { 
        (ref.page_idx, ref.block_index, ref.image_path)
        : _build_summary(screening_result, understanding_result)
        for ref, (screening_result, understanding_result) in results.items() 
    }

    middle = middle.model_copy(deep=True)
    for page in middle.pdf_info:
        for block in page.para_blocks:
            _merge_understanding_into_block(
                block,
                page_idx=page.page_idx,
                dict= dict,
            )
        for block in page.discarded_blocks:
            _merge_understanding_into_block(
                block,
                page_idx=page.page_idx,
                dict= dict,
            )
        for block in page.preproc_blocks:
            _merge_understanding_into_block(
                block,
                page_idx=page.page_idx,
                dict= dict,
            )
    return middle


def _build_summary(
    screening: ScreeningResult, 
    understanding: UnderstandingResult
) -> ImageUnderstandingSummary:
    return ImageUnderstandingSummary(
        kind= screening.kind,
        rag_value= screening.rag_value,
        detail_level= screening.detail_level,
        keywords= understanding.keywords,
        notes= understanding.notes,
        short_description= understanding.short_description,
        long_description= understanding.long_description,
    )


def _merge_understanding_into_block(
    block: Block,
    *,
    page_idx: int,
    dict : dict[tuple[int, int | None, str], ImageUnderstandingSummary],
    owner_block_index: int | None = None,
) -> None:
    current_block_index = block.index if block.index is not None else owner_block_index
    for line in block.lines:
        for span in line.spans:
            _merge_understanding_into_span(
                span,
                page_idx=page_idx,
                block_index=current_block_index,
                dict= dict,
            )
    for child in block.blocks:
        _merge_understanding_into_block(
            child,
            page_idx=page_idx,
            dict= dict,
            owner_block_index=current_block_index,
        )


def _merge_understanding_into_span(
    span: Span,
    *,
    page_idx: int,
    block_index: int | None,
    dict : dict[tuple[int, int | None, str], ImageUnderstandingSummary],
) -> None:
    if span.image_path is None:
        return
    if summary := dict.get((page_idx, block_index, span.image_path)):
        span.image_understanding = summary
    else:
        logger.warning('Image not analyzed: %s', (page_idx, block_index, span.image_path))
