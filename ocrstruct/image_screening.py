from __future__ import annotations

import base64
import json
import logging

import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Literal, TypeAlias, cast

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ocrstruct.middle import Block, ImageUnderstandingSummary, Middle, Span, normalize_text
from ocrstruct.utils import BaseModelWithSave, sha256_file
from ocrstruct.result import Parameters
from ocrstruct.vlm import (
    TokenUsage, PriceEstimate, ModelPricing, VLM, VLMConfig, 
    DEFAULT_MODEL_PRICING,
    image_json_request, 
    image_data_url,
    estimate_price_from_completion
)


ImageBlockType: TypeAlias = Literal["image", "chart", "seal"]

ImageKind: TypeAlias = Literal[
    "diagram",
    "table_or_form",
    "chart_or_graph",
    "ui_or_screenshot",
    "arrow_only",
    "code_symbol",
    "seal",
    "text_as_image",
    "decorative",
    "logo_or_mark",
    "other",
]

RagValue: TypeAlias = Literal["high", "medium", "low", "none"]

DetailLevel: TypeAlias = Literal["skip", "short", "long"]


class ImageRef(BaseModel, frozen= True):
    pdf_path: str
    outdir: str
    page_idx: int
    block_index: int | None = None
    block_type: ImageBlockType
    image_path: str
    caption: str | None = None
    nearby_text_before: str | None = None
    nearby_text_after: str | None = None
    section_title: str | None = None

    @property
    def path(self) -> Path:
        image_path = Path(self.image_path)
        if image_path.is_absolute():
            return image_path
        return Path(self.outdir) / "images" / image_path

    @property
    def image_data_url(self) -> str:
        return image_data_url(self.path)


class ScreeningResult(BaseModel, frozen= True):
    kind: ImageKind
    rag_value: RagValue
    detail_level: DetailLevel
    notes: str | None = None


class ScreeningRunResult(BaseModel):
    vlm: VLM
    started_at: str
    raw_text: str
    result: ScreeningResult
    usage: TokenUsage | None = None
    price: PriceEstimate | None = None


CONTEXT_BLOCK_TYPES = {
    "text",
    "abstract",
    "list",
    "index",
    "ref_text",
    "aside_text",
    "page_footnote",
}

TITLE_BLOCK_TYPES = {
    "title",
    "doc_title",
    "paragraph_title",
}

MAX_NEARBY_TEXT_CHARS = 300
MIN_NEARBY_TEXT_CHARS = 200


def _span_text(block: Block) -> str:
    parts: list[str] = []
    for line in block.lines:
        line_parts: list[str] = []
        for span in line.spans:
            content = span.content
            if content is None:
                continue
            if isinstance(content, str):
                line_parts.append(content)
            else:
                line_parts.append("".join(content))
        if line_parts:
            parts.append("".join(line_parts))
    if not parts:
        return ""
    return normalize_text("\n".join(parts))


def _caption_text(block: Block) -> str | None:
    parts: list[str] = []
    for child in block.blocks:
        if child.type.endswith("_caption") or child.type == "caption":
            text = _span_text(child)
            if text:
                parts.append(text)
    if not parts:
        return None
    return normalize_text("\n".join(parts))


def _truncate_text(text: str, *, max_chars: int = MAX_NEARBY_TEXT_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _context_text(block: Block) -> str | None:
    text = _span_text(block)
    if not text:
        return None
    return _truncate_text(text)


def _combined_context_text(blocks: list[Block]) -> str | None:
    parts: list[str] = []
    total_chars = 0
    for block in blocks:
        text = _span_text(block)
        if not text:
            continue
        parts.append(text)
        total_chars += len(text)
        if total_chars >= MIN_NEARBY_TEXT_CHARS:
            break
    if not parts:
        return None
    return _truncate_text("\n".join(parts))


def image_refs_from_middle(
    middle: Middle,
    *,
    pdf_path: str,
    outdir: str,
) -> list[ImageRef]:
    '''Extract images and their contexts from Middle'''
    out: list[ImageRef] = []
    for page in middle.pdf_info:
        blocks = page.para_blocks
        for i, block in enumerate(blocks):
            if block.type not in ("image", "chart", "seal"):
                continue
            caption = _caption_text(block)
            block_type: ImageBlockType = block.type
            section_title: str | None = None
            nearby_text_before: str | None = None
            nearby_text_after: str | None = None

            before_context_blocks: list[Block] = []
            for prev in reversed(blocks[:i]):
                if section_title is None and prev.type in TITLE_BLOCK_TYPES:
                    section_title = _context_text(prev)
                if prev.type in CONTEXT_BLOCK_TYPES:
                    before_context_blocks.append(prev)
                if section_title is not None and before_context_blocks:
                    break
            before_context_blocks.reverse()
            nearby_text_before = _combined_context_text(before_context_blocks)

            after_context_blocks: list[Block] = []
            for next_block in blocks[i + 1 :]:
                if next_block.type in CONTEXT_BLOCK_TYPES:
                    after_context_blocks.append(next_block)
                    text = _span_text(next_block)
                    if text and sum(len(_span_text(block)) for block in after_context_blocks) >= MIN_NEARBY_TEXT_CHARS:
                        break
            nearby_text_after = _combined_context_text(after_context_blocks)

            for child in block.blocks:
                for line in child.lines:
                    for span in line.spans:
                        if span.image_path is None:
                            continue
                        out.append(
                            ImageRef(
                                pdf_path=pdf_path,
                                outdir= outdir,
                                page_idx=page.page_idx,
                                block_index=block.index,
                                block_type=block_type,
                                image_path=span.image_path,
                                caption=caption,
                                nearby_text_before=nearby_text_before,
                                nearby_text_after=nearby_text_after,
                                section_title=section_title,
                            )
                        )
    return out


def screening_context_text(ref: ImageRef) -> str:
    parts = [
        f"PDF path: {ref.pdf_path}",
        f"Out directory: {ref.outdir}",
        f"Page index: {ref.page_idx}",
        f"Block index: {ref.block_index}",
        f"Block type: {ref.block_type}",
    ]
    if ref.section_title is not None:
        parts.append(f"Section title: {ref.section_title}")
    if ref.caption is not None:
        parts.append(f"Caption: {ref.caption}")
    if ref.nearby_text_before is not None:
        parts.append(f"Nearby text before: {ref.nearby_text_before}")
    if ref.nearby_text_after is not None:
        parts.append(f"Nearby text after: {ref.nearby_text_after}")
    return "\n".join(parts)


def run_image_screening(
    vlm_config: VLMConfig,
    ref: ImageRef,
) -> ScreeningRunResult:
    client = OpenAI(api_key=vlm_config.api_key, base_url=vlm_config.base_url)
    started_at = datetime.now(UTC).isoformat()
    prompt = """
You classify document images for RAG enrichment.

Return strict JSON with this shape:
{
  "kind": "diagram" | "table_or_form" | "chart_or_graph" | "ui_or_screenshot" | "arrow_only" | "code_symbol" | "seal" | "text_as_image" | "decorative" | "logo_or_mark" | "other",
  "rag_value": "high" | "medium" | "low" | "none",
  "detail_level": "skip" | "short" | "long",
  "notes": string | null
}

Guidance:
- Prefer "diagram" for flowchart-like process figures, relationship diagrams, and schematics.
- Prefer "table_or_form" for tables, matrices, checklists, and form-like layouts.
- Prefer "chart_or_graph" for plots and quantitative charts such as bar, line, or pie charts.
- Prefer "ui_or_screenshot" for screenshots of software, websites, or UI walkthroughs.
- Prefer "code_symbol" for standalone QR codes, barcodes, and similar machine-readable symbols.
- Prefer "seal" for stamps, official seals, hanko-like marks, and similar印影.
- Prefer "text_as_image" when the image is mostly text that should likely be read.
- Prefer "arrow_only" for arrows or nearly content-free directional marks.
- Prefer "decorative" for photos or icons with little RAG value.
- Prefer "code_symbol" for QR/barcode.
- Prefer "decorative" for standalone photos.
- Use "other" only as a last resort.
- Keep "notes" concise: at most 40 Japanese characters when written in Japanese.
- "notes" should briefly state the reason for the classification.
- Use null only when the image is trivial and the classification is obvious.
- "detail_level" is cumulative: long implies short.
- Use "skip" for images with little or no RAG value.
""".strip()
    request = image_json_request(
        vlm= vlm_config.vlm,
        prompt=prompt,
        context_text= screening_context_text(ref),
        image_data_url= ref.image_data_url,
        schema_name="screening_result",
        schema_model=ScreeningResult,
    )
    last_error: Exception | None = None
    for _attempt in range(1):
        completion = client.chat.completions.create(**cast(Any, request))
        content = completion.choices[0].message.content
        if content is None:
            last_error = ValueError("Model returned no content")
            continue
        try:
            result = ScreeningResult.model_validate_json(content)
            pricing = (
                vlm_config.pricing 
                if vlm_config.pricing is not None else DEFAULT_MODEL_PRICING.get(vlm_config.model)
            )
            usage, price = estimate_price_from_completion(completion, pricing)
            return ScreeningRunResult(
                vlm= VLM(model= vlm_config.model, thinking= vlm_config.thinking),
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
