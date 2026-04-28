from __future__ import annotations

import base64
import json
import mimetypes
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, TypeAlias

from openai import OpenAI
from pydantic import ConfigDict

from ocrstruct.middle import Block, Middle, normalize_text
from ocrstruct.utils import BaseModelWithSave


ImageBlockType: TypeAlias = Literal["image", "chart", "seal"]

ImageKind: TypeAlias = Literal[
    "process_diagram",
    "schematic",
    "arrow_only",
    "text_as_image",
    "decorative_photo",
    "decorative_icon",
    "logo_or_mark",
    "other",
]

RagValue: TypeAlias = Literal["high", "medium", "low", "none"]

DetailLevel: TypeAlias = Literal["skip", "short", "long", "extract_text"]


class Model(BaseModelWithSave):
    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
        serialize_by_alias=True,
    )


class ImageRef(Model):
    pdf_path: str
    middle_json_path: str
    page_idx: int
    block_index: int | None = None
    block_type: ImageBlockType
    image_path: str
    caption: str | None = None
    nearby_text_before: str | None = None
    nearby_text_after: str | None = None
    section_title: str | None = None


class ScreeningResult(Model):
    kind: ImageKind
    rag_value: RagValue
    detail_level: DetailLevel
    notes: str | None = None


class TokenUsage(Model):
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


class PriceEstimate(Model):
    input_cost_usd: float | None = None
    output_cost_usd: float | None = None
    total_cost_usd: float | None = None


class ModelPricing(Model):
    input_per_million_usd: float
    output_per_million_usd: float


class ScreeningRunResult(Model):
    model: str
    base_url: str | None = None
    started_at: str
    raw_text: str
    result: ScreeningResult
    usage: TokenUsage | None = None
    price: PriceEstimate | None = None


class ImageUnderstanding(Model):
    ref: ImageRef
    kind: ImageKind
    rag_value: RagValue
    detail_level: DetailLevel
    notes: str | None = None
    short_description: str | None = None
    long_description: str | None = None
    extracted_text: str | None = None


class ImageUnderstandingFile(Model):
    items: list[ImageUnderstanding]


DEFAULT_MODEL_PRICING: dict[str, ModelPricing] = {
    "gpt-5": ModelPricing(input_per_million_usd=1.25, output_per_million_usd=10.0),
    "gpt-5-mini": ModelPricing(input_per_million_usd=0.25, output_per_million_usd=2.0),
    "gpt-5-nano": ModelPricing(input_per_million_usd=0.05, output_per_million_usd=0.4),
    "gpt-4.1": ModelPricing(input_per_million_usd=2.0, output_per_million_usd=8.0),
    "gpt-4.1-mini": ModelPricing(input_per_million_usd=0.4, output_per_million_usd=1.6),
    "gpt-4o": ModelPricing(input_per_million_usd=2.5, output_per_million_usd=10.0),
    "gpt-4o-mini": ModelPricing(input_per_million_usd=0.15, output_per_million_usd=0.6),
}


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


def image_refs_from_middle(
    middle: Middle,
    *,
    pdf_path: str,
    middle_json_path: str,
) -> list[ImageRef]:
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

            for prev in reversed(blocks[:i]):
                if section_title is None and prev.type in TITLE_BLOCK_TYPES:
                    section_title = _context_text(prev)
                if nearby_text_before is None and prev.type in CONTEXT_BLOCK_TYPES:
                    nearby_text_before = _context_text(prev)
                if section_title is not None and nearby_text_before is not None:
                    break

            for next_block in blocks[i + 1 :]:
                if next_block.type in CONTEXT_BLOCK_TYPES:
                    nearby_text_after = _context_text(next_block)
                    if nearby_text_after is not None:
                        break

            for child in block.blocks:
                for line in child.lines:
                    for span in line.spans:
                        if span.image_path is None:
                            continue
                        out.append(
                            ImageRef(
                                pdf_path=pdf_path,
                                middle_json_path=middle_json_path,
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


def _image_file_path(ref: ImageRef) -> Path:
    image_path = Path(ref.image_path)
    if image_path.is_absolute():
        return image_path
    return Path(ref.middle_json_path).parent / "images" / image_path


def _image_data_url(ref: ImageRef) -> str:
    image_file = _image_file_path(ref)
    mime_type, _encoding = mimetypes.guess_type(image_file.name)
    if mime_type is None:
        mime_type = "application/octet-stream"
    raw = image_file.read_bytes()
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _screening_context_text(ref: ImageRef) -> str:
    parts = [
        f"PDF path: {ref.pdf_path}",
        f"Middle path: {ref.middle_json_path}",
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


def _usage_from_completion(completion: object) -> TokenUsage | None:
    usage = getattr(completion, "usage", None)
    if usage is None:
        return None
    input_tokens = getattr(usage, "prompt_tokens", None)
    output_tokens = getattr(usage, "completion_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )


def estimate_price(
    usage: TokenUsage | None,
    pricing: ModelPricing | None,
) -> PriceEstimate | None:
    if usage is None or pricing is None:
        return None
    input_cost_usd: float | None = None
    output_cost_usd: float | None = None
    if usage.input_tokens is not None:
        input_cost_usd = usage.input_tokens * pricing.input_per_million_usd / 1_000_000
    if usage.output_tokens is not None:
        output_cost_usd = usage.output_tokens * pricing.output_per_million_usd / 1_000_000
    if input_cost_usd is None and output_cost_usd is None:
        return None
    total_cost_usd = (input_cost_usd or 0.0) + (output_cost_usd or 0.0)
    return PriceEstimate(
        input_cost_usd=input_cost_usd,
        output_cost_usd=output_cost_usd,
        total_cost_usd=total_cost_usd,
    )


def screening_run_from_image_ref(
    ref: ImageRef,
    *,
    model: str,
    base_url: str | None = None,
    api_key: str | None = None,
    pricing: ModelPricing | None = None,
) -> ScreeningRunResult:
    client = OpenAI(
        api_key=api_key if api_key is not None else os.environ.get("OPENAI_API_KEY"),
        base_url=base_url,
    )
    started_at = datetime.now(UTC).isoformat()
    prompt = """
You classify document images for RAG enrichment.

Return strict JSON with this shape:
{
  "kind": "process_diagram" | "schematic" | "arrow_only" | "text_as_image" | "decorative_photo" | "decorative_icon" | "logo_or_mark" | "other",
  "rag_value": "high" | "medium" | "low" | "none",
  "detail_level": "skip" | "short" | "long" | "extract_text",
  "notes": string | null
}

Guidance:
- Prefer "process_diagram" for flowchart-like process figures.
- Prefer "text_as_image" when the image is mostly text that should likely be read.
- Prefer "arrow_only" for arrows or nearly content-free directional marks.
- Prefer "decorative_photo" or "decorative_icon" for images with little RAG value.
- "detail_level" is cumulative: long implies short, extract_text implies long.
- Use "skip" for images with little or no RAG value.
- Use "extract_text" only when careful text extraction seems worthwhile.
""".strip()
    completion = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": _screening_context_text(ref),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": _image_data_url(ref)},
                    },
                ],
            },
        ],
    )
    content = completion.choices[0].message.content
    if content is None:
        raise ValueError("Model returned no content")
    result = ScreeningResult.model_validate(json.loads(content))
    usage = _usage_from_completion(completion)
    price = estimate_price(usage, pricing if pricing is not None else DEFAULT_MODEL_PRICING.get(model))
    return ScreeningRunResult(
        model=model,
        base_url=base_url,
        started_at=started_at,
        raw_text=content,
        result=result,
        usage=usage,
        price=price,
    )


def screening_result_from_image_ref(
    ref: ImageRef,
    *,
    model: str,
    base_url: str | None = None,
    api_key: str | None = None,
) -> ScreeningResult:
    return screening_run_from_image_ref(
        ref,
        model=model,
        base_url=base_url,
        api_key=api_key,
    ).result
