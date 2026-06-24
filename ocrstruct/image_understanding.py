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
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ocrstruct.middle import Block, ImageUnderstandingSummary, Middle, Span, normalize_text
from ocrstruct.utils import BaseModelWithSave, sha256_file


logger = logging.getLogger(__name__)


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


class ImageUnderstandingRunResult(Model):
    model: str
    base_url: str | None = None
    started_at: str
    raw_text: str
    result: ImageUnderstanding
    usage: TokenUsage | None = None
    price: PriceEstimate | None = None


class ImageUnderstanding(Model):
    ref: ImageRef
    kind: ImageKind
    rag_value: RagValue
    detail_level: DetailLevel
    keywords: list[str] = Field(default_factory=list)
    notes: str | None = None
    short_description: str | None = None
    long_description: str | None = None


class ImageResultRef(Model):
    page_idx: int
    block_index: int | None = None
    block_type: ImageBlockType
    image_path: str
    caption: str | None = None
    nearby_text_before: str | None = None
    nearby_text_after: str | None = None
    section_title: str | None = None


class FinalScreeningResult(Model):
    model: str
    thinking: bool | None = None
    resolved_thinking: bool | None = None
    base_url: str | None = None
    started_at: str | None = None
    latency_sec: float
    usage: TokenUsage | None = None
    price: PriceEstimate | None = None
    kind: ImageKind
    rag_value: RagValue
    detail_level: DetailLevel
    notes: str | None = None
    raw_text: str


class FinalUnderstandingResult(FinalScreeningResult):
    keywords: list[str] = Field(default_factory=list)
    short_description: str | None = None
    long_description: str | None = None


class ImageResult(Model):
    ref: ImageResultRef
    screening: FinalScreeningResult
    understanding: FinalUnderstandingResult


class ImagesFile(Model):
    version: Literal["1"] = "1"
    middle_json_sha256: str
    items: list[ImageResult] = Field(default_factory=list)


class StructuredOutputModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ScreeningPayload(StructuredOutputModel):
    kind: ImageKind
    rag_value: RagValue
    detail_level: DetailLevel
    notes: str | None = None


class UnderstandingShortPayload(StructuredOutputModel):
    keywords: list[str]
    short_description: str
    notes: str | None = None


class UnderstandingLongPayload(StructuredOutputModel):
    keywords: list[str]
    short_description: str
    long_description: str
    notes: str | None = None


class RunStatus(Model):
    ok: bool
    error: str | None = None


class ScreeningRunView(Model):
    kind: ImageKind
    rag_value: RagValue
    detail_level: DetailLevel
    notes: str | None = None
    raw_text: str
    usage: TokenUsage | None = None
    price: PriceEstimate | None = None

    def to_screening_result(self) -> ScreeningResult:
        return ScreeningResult(
            kind=self.kind,
            rag_value=self.rag_value,
            detail_level=self.detail_level,
            notes=self.notes,
        )


class ScreeningRecord(Model):
    ref: ImageRef
    model: str
    thinking: bool | None = None
    resolved_thinking: bool | None = None
    base_url: str | None = None
    started_at: str | None = None
    latency_sec: float
    status: RunStatus
    run: ScreeningRunView | None = None


class UnderstandingRunView(Model):
    kind: str
    rag_value: str
    detail_level: str
    keywords: list[str] = Field(default_factory=list)
    notes: str | None = None
    short_description: str | None = None
    long_description: str | None = None
    raw_text: str
    usage: TokenUsage | None = None
    price: PriceEstimate | None = None

    @classmethod
    def from_image_understanding_run_result(
        cls,
        run: ImageUnderstandingRunResult,
    ) -> "UnderstandingRunView":
        return cls(
            kind=run.result.kind,
            rag_value=run.result.rag_value,
            detail_level=run.result.detail_level,
            keywords=run.result.keywords,
            notes=run.result.notes,
            short_description=run.result.short_description,
            long_description=run.result.long_description,
            raw_text=run.raw_text,
            usage=run.usage,
            price=run.price,
        )


class ScreeningSource(Model):
    model: str
    thinking: bool | None = None
    resolved_thinking: bool | None = None
    base_url: str | None = None
    started_at: str | None = None
    latency_sec: float
    run: ScreeningRunView


class UnderstandingRecord(Model):
    ref: ImageRef
    screening: ScreeningSource
    model: str
    thinking: bool = False
    resolved_thinking: bool = False
    base_url: str | None = None
    started_at: str | None = None
    latency_sec: float
    status: RunStatus
    run: UnderstandingRunView | None = None


DEFAULT_MODEL_PRICING: dict[str, ModelPricing] = {
    "Qwen/Qwen3.6-35B-A3B-FP8": ModelPricing(
        input_per_million_usd=0.0,
        output_per_million_usd=0.0,
    ),
    "Qwen/Qwen3.6-27B-FP8": ModelPricing(
        input_per_million_usd=0.0,
        output_per_million_usd=0.0,
    ),
    "google/gemma-4-26B-A4B-it": ModelPricing(
        input_per_million_usd=0.0,
        output_per_million_usd=0.0,
    ),
    "gpt-5.4": ModelPricing(input_per_million_usd=2.5, output_per_million_usd=15.0),
    "gpt-5.4-mini": ModelPricing(input_per_million_usd=0.75, output_per_million_usd=4.5),
    "gpt-5.4-nano": ModelPricing(input_per_million_usd=0.2, output_per_million_usd=1.25),
    "gpt-5.2": ModelPricing(input_per_million_usd=1.75, output_per_million_usd=14.0),
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


def _understanding_context_text(ref: ImageRef, screening: ScreeningResult) -> str:
    parts = [
        _screening_context_text(ref),
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


def _apply_thinking_option(
    request: dict[str, object],
    *,
    model: str,
    thinking: bool = False,
) -> None:
    supports_chat_template_thinking = model.startswith(("Qwen/", "google/gemma-4"))
    if thinking is True:
        if model.startswith(("gpt-5.4", "gpt-5.5", "gpt-5.1", "gpt-5")):
            request["reasoning_effort"] = "medium"
        elif supports_chat_template_thinking:
            request["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": True},
            }
    elif thinking is False:
        if model.startswith(("gpt-5.4", "gpt-5.5", "gpt-5.1")):
            request["reasoning_effort"] = "none"
        elif model.startswith("gpt-5"):
            request["reasoning_effort"] = "minimal"
        elif supports_chat_template_thinking:
            request["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": False},
            }


def _openai_strict_json_schema(schema_model: type[BaseModel]) -> dict[str, object]:
    schema = cast(dict[str, object], schema_model.model_json_schema())
    properties = cast(dict[str, object], schema.get("properties", {}))
    if properties:
        schema["required"] = list(properties.keys())
    return schema


def _image_json_request(
    *,
    model: str,
    prompt: str,
    context_text: str,
    ref: ImageRef,
    thinking: bool,
    schema_name: str,
    schema_model: type[BaseModel],
) -> dict[str, object]:
    request: dict[str, object] = {
        "model": model,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": _openai_strict_json_schema(schema_model),
            },
        },
        "messages": [
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": context_text,
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": _image_data_url(ref)},
                    },
                ],
            },
        ],
    }
    _apply_thinking_option(request, model=model, thinking=thinking)
    return request


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
    thinking: bool = False,
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
    request = _image_json_request(
        model=model,
        prompt=prompt,
        context_text=_screening_context_text(ref),
        ref=ref,
        thinking=thinking,
        schema_name="screening_result",
        schema_model=ScreeningPayload,
    )
    last_error: Exception | None = None
    for _attempt in range(1):
        completion = client.chat.completions.create(**cast(Any, request))
        content = completion.choices[0].message.content
        if content is None:
            last_error = ValueError("Model returned no content")
            continue
        try:
            payload = ScreeningPayload.model_validate_json(content)
            result = ScreeningResult.model_validate(payload.model_dump())
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
        except (ValidationError, ValueError, json.JSONDecodeError) as error:
            last_error = error
            continue
    if last_error is None:
        raise ValueError("Model returned no content")
    raise last_error


def _image_understanding_from_payload(
    ref: ImageRef,
    screening: ScreeningResult,
    payload: dict[str, object],
) -> ImageUnderstanding:
    keywords = payload.get("keywords")
    short_description = payload.get("short_description")
    long_description = payload.get("long_description")
    notes = payload.get("notes")
    return ImageUnderstanding(
        ref=ref,
        kind=screening.kind,
        rag_value=screening.rag_value,
        detail_level=screening.detail_level,
        keywords=cast(list[str], keywords if keywords is not None else []),
        notes=cast(str | None, notes),
        short_description=cast(str | None, short_description),
        long_description=cast(str | None, long_description),
    )


def _skip_image_understanding_result(
    ref: ImageRef,
    screening: ScreeningResult,
    *,
    model: str,
    base_url: str | None = None,
) -> ImageUnderstandingRunResult:
    started_at = datetime.now(UTC).isoformat()
    payload = {
        "keywords": [],
        "short_description": None,
        "long_description": None,
        "notes": screening.notes,
    }
    return ImageUnderstandingRunResult(
        model=model,
        base_url=base_url,
        started_at=started_at,
        raw_text=json.dumps(payload, ensure_ascii=False),
        result=_image_understanding_from_payload(ref, screening, payload),
        usage=None,
        price=PriceEstimate(
            input_cost_usd=0.0,
            output_cost_usd=0.0,
            total_cost_usd=0.0,
        ),
    )


def _understanding_payload_model(
    detail_level: DetailLevel,
) -> type[StructuredOutputModel]:
    match detail_level:
        case "short":
            return UnderstandingShortPayload
        case "long":
            return UnderstandingLongPayload
        case _:
            raise ValueError(f"Unsupported detail_level for understanding payload: {detail_level}")


def image_understanding_run_from_screening(
    ref: ImageRef,
    screening: ScreeningResult,
    *,
    model: str,
    base_url: str | None = None,
    api_key: str | None = None,
    pricing: ModelPricing,
    thinking: bool = False,
) -> ImageUnderstandingRunResult:
    if screening.detail_level == "skip":
        return _skip_image_understanding_result(
            ref,
            screening,
            model=model,
            base_url=base_url,
        )

    client = OpenAI(
        api_key=api_key if api_key is not None else os.environ.get("OPENAI_API_KEY"),
        base_url=base_url,
    )
    started_at = datetime.now(UTC).isoformat()
    prompt = _image_understanding_prompt(screening.kind, screening.detail_level)
    request = _image_json_request(
        model=model,
        prompt=prompt,
        context_text=_understanding_context_text(ref, screening),
        ref=ref,
        thinking=thinking,
        schema_name=f"image_understanding_{screening.detail_level}",
        schema_model=_understanding_payload_model(screening.detail_level),
    )
    payload_model = _understanding_payload_model(screening.detail_level)
    last_error: Exception | None = None
    for _attempt in range(1):
        completion = client.chat.completions.create(**cast(Any, request))
        content = completion.choices[0].message.content
        if content is None:
            last_error = ValueError("Model returned no content")
            continue
        try:
            payload_obj = payload_model.model_validate_json(content)
            payload = cast(dict[str, object], payload_obj.model_dump())
            result = _image_understanding_from_payload(ref, screening, payload)
            usage = _usage_from_completion(completion)
            price = estimate_price(usage, pricing)
            return ImageUnderstandingRunResult(
                model=model,
                base_url=base_url,
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


UnderstandingRecordKey: TypeAlias = tuple[
    str,
    bool,
    str,
    bool | None,
    tuple[str, int, int | None, str],
]

ScreeningRecordKey: TypeAlias = tuple[
    str,
    bool | None,
    tuple[str, int, int | None, str],
]


def load_pricing_overrides(path: str | Path | None) -> dict[str, ModelPricing]:
    if path is None:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Pricing override must be a JSON object: {path}")
    out: dict[str, ModelPricing] = {}
    for model, value in data.items():
        out[model] = ModelPricing.model_validate(value)
    return out


def pricing_for_model(
    model: str,
    pricing_overrides: dict[str, ModelPricing] | None = None,
) -> ModelPricing:
    overrides = pricing_overrides or {}
    pricing = overrides.get(model, DEFAULT_MODEL_PRICING.get(model))
    if pricing is None:
        raise ValueError(
            "Unknown model pricing: "
            f"{model}. Add it to DEFAULT_MODEL_PRICING or pass --pricing-json."
        )
    return pricing


def screening_record_ref_key(ref: ImageRef) -> tuple[str, int, int | None, str]:
    return (
        ref.middle_json_path,
        ref.page_idx,
        ref.block_index,
        ref.image_path,
    )


def image_result_ref_key(ref: ImageResultRef) -> tuple[int, int | None, str]:
    return (
        ref.page_idx,
        ref.block_index,
        ref.image_path,
    )


def image_result_ref_from_image_ref(ref: ImageRef) -> ImageResultRef:
    return ImageResultRef(
        page_idx=ref.page_idx,
        block_index=ref.block_index,
        block_type=ref.block_type,
        image_path=ref.image_path,
        caption=ref.caption,
        nearby_text_before=ref.nearby_text_before,
        nearby_text_after=ref.nearby_text_after,
        section_title=ref.section_title,
    )


def image_result_from_understanding_record(record: UnderstandingRecord) -> ImageResult:
    if not record.status.ok or record.run is None:
        raise ValueError("UnderstandingRecord must be successful to convert to ImageResult")
    screening_run = record.screening.run
    return ImageResult(
        ref=image_result_ref_from_image_ref(record.ref),
        screening=FinalScreeningResult(
            model=record.screening.model,
            thinking=record.screening.thinking,
            resolved_thinking=record.screening.resolved_thinking,
            base_url=record.screening.base_url,
            started_at=record.screening.started_at,
            latency_sec=record.screening.latency_sec,
            usage=screening_run.usage,
            price=screening_run.price,
            kind=screening_run.kind,
            rag_value=screening_run.rag_value,
            detail_level=screening_run.detail_level,
            notes=screening_run.notes,
            raw_text=screening_run.raw_text,
        ),
        understanding=FinalUnderstandingResult(
            model=record.model,
            thinking=record.thinking,
            resolved_thinking=record.resolved_thinking,
            base_url=record.base_url,
            started_at=record.started_at,
            latency_sec=record.latency_sec,
            usage=record.run.usage,
            price=record.run.price,
            kind=cast(ImageKind, record.run.kind),
            rag_value=cast(RagValue, record.run.rag_value),
            detail_level=cast(DetailLevel, record.run.detail_level),
            keywords=record.run.keywords,
            notes=record.run.notes,
            short_description=record.run.short_description,
            long_description=record.run.long_description,
            raw_text=record.run.raw_text,
        ),
    )


def build_images_file(
    records: Iterable[UnderstandingRecord],
    *,
    middle_json_path: str | Path,
) -> ImagesFile:
    items = [
        image_result_from_understanding_record(record)
        for record in records
    ]
    items.sort(
        key=lambda item: (
            item.ref.page_idx,
            -1 if item.ref.block_index is None else item.ref.block_index,
            item.ref.image_path,
        )
    )
    return ImagesFile(
        middle_json_sha256=sha256_file(middle_json_path),
        items=items,
    )


def load_images_file_json(
    path: str | Path,
    *,
    middle_json_sha256: str | None = None,
    middle_json_path: str | Path | None = None,
) -> ImagesFile:
    images_file = ImagesFile.load_json(path)
    expected_sha256 = middle_json_sha256
    if middle_json_path is not None:
        computed_sha256 = sha256_file(middle_json_path)
        if expected_sha256 is not None and expected_sha256 != computed_sha256:
            raise ValueError("middle_json_sha256 does not match the supplied middle_json_path")
        expected_sha256 = computed_sha256
    if expected_sha256 is not None and images_file.middle_json_sha256 != expected_sha256:
        raise ValueError("images.json does not match the requested middle.json hash")
    return images_file


def screening_record_key(record: ScreeningRecord) -> ScreeningRecordKey:
    return (record.model, record.thinking, screening_record_ref_key(record.ref))


def load_completed_screening_keys(path: str | Path) -> set[ScreeningRecordKey]:
    target = Path(path)
    if not target.exists():
        return set()
    completed: set[ScreeningRecordKey] = set()
    for line in target.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = ScreeningRecord.model_validate_json(line)
        if not record.status.ok:
            continue
        completed.add(screening_record_key(record))
    return completed


def iter_screening_records_from_refs(
    refs: Iterable[ImageRef],
    *,
    model: str,
    pricing: ModelPricing,
    base_url: str | None = None,
    api_key: str | None = None,
    thinking: bool = False,
    existing_keys: set[ScreeningRecordKey] | None = None,
) -> Iterable[ScreeningRecord]:
    completed_keys = existing_keys if existing_keys is not None else set()
    items = list(refs)
    total = len(items)

    for index, ref in enumerate(items, start=1):
        completed_key = (model, thinking, screening_record_ref_key(ref))
        if completed_key in completed_keys:
            logger.info(
                "Skipping existing model=%s thinking=%s item=%d/%d image=%s",
                model,
                thinking,
                index,
                total,
                ref.image_path,
            )
            continue

        logger.info(
            "Processing model=%s thinking=%s item=%d/%d image=%s",
            model,
            thinking,
            index,
            total,
            ref.image_path,
        )
        started = time.perf_counter()
        try:
            run = screening_run_from_image_ref(
                ref,
                model=model,
                base_url=base_url,
                api_key=api_key,
                pricing=pricing,
                thinking=thinking,
            )
            record = ScreeningRecord(
                ref=ref,
                model=model,
                thinking=thinking,
                resolved_thinking=thinking,
                base_url=base_url,
                started_at=run.started_at,
                latency_sec=time.perf_counter() - started,
                status=RunStatus(ok=True),
                run=ScreeningRunView(
                    kind=run.result.kind,
                    rag_value=run.result.rag_value,
                    detail_level=run.result.detail_level,
                    notes=run.result.notes,
                    raw_text=run.raw_text,
                    usage=run.usage,
                    price=run.price,
                ),
            )
            logger.info(
                "Completed model=%s thinking=%s item=%d/%d image=%s latency=%.2fs",
                model,
                thinking,
                index,
                total,
                ref.image_path,
                record.latency_sec,
            )
        except (OSError, ValueError, ValidationError) as error:
            record = ScreeningRecord(
                ref=ref,
                model=model,
                thinking=thinking,
                resolved_thinking=thinking,
                base_url=base_url,
                started_at=None,
                latency_sec=time.perf_counter() - started,
                status=RunStatus(ok=False, error=str(error)),
                run=None,
            )
            logger.exception(
                "Failed model=%s thinking=%s item=%d/%d image=%s",
                model,
                thinking,
                index,
                total,
                ref.image_path,
            )
        yield record


def understanding_record_key(record: UnderstandingRecord) -> UnderstandingRecordKey:
    return (
        record.model,
        record.thinking,
        record.screening.model,
        record.screening.thinking,
        screening_record_ref_key(record.ref),
    )


def load_screening_records_jsonl(
    path: str | Path,
    *,
    screening_thinking: bool | None = None,
) -> list[ScreeningRecord]:
    out: list[ScreeningRecord] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = ScreeningRecord.model_validate_json(line)
        if not record.status.ok or record.run is None:
            continue
        if screening_thinking is not None and record.thinking != screening_thinking:
            continue
        out.append(record)
    return out


def load_completed_understanding_keys(path: str | Path) -> set[UnderstandingRecordKey]:
    target = Path(path)
    if not target.exists():
        return set()
    completed: set[UnderstandingRecordKey] = set()
    for line in target.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = UnderstandingRecord.model_validate_json(line)
        if not record.status.ok:
            continue
        completed.add(understanding_record_key(record))
    return completed


def load_understanding_records_jsonl(
    path: str | Path,
    *,
    only_success: bool = True,
) -> list[UnderstandingRecord]:
    out: list[UnderstandingRecord] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = UnderstandingRecord.model_validate_json(line)
        if only_success and (not record.status.ok or record.run is None):
            continue
        out.append(record)
    return out


def merge_understanding_into_middle(
    middle: Middle,
    records: Iterable[UnderstandingRecord],
) -> Middle:
    summary_by_ref: dict[tuple[int, int | None, str], ImageUnderstandingSummary] = {}
    for record in records:
        if not record.status.ok or record.run is None:
            continue
        summary_by_ref[(
            record.ref.page_idx,
            record.ref.block_index,
            record.ref.image_path,
        )] = ImageUnderstandingSummary(
            kind=record.run.kind,
            rag_value=record.run.rag_value,
            detail_level=record.run.detail_level,
            keywords=record.run.keywords,
            notes=record.run.notes,
            short_description=record.run.short_description,
            long_description=record.run.long_description,
            model=record.model,
            thinking=record.thinking,
            screening_model=record.screening.model,
            screening_thinking=record.screening.thinking,
            status_ok=record.status.ok,
        )

    return _merge_image_summaries_into_middle(
        middle,
        summary_by_ref,
    )


def merge_images_into_middle(
    middle: Middle,
    images_file: ImagesFile,
) -> Middle:
    summary_by_ref: dict[tuple[int, int | None, str], ImageUnderstandingSummary] = {}
    for item in images_file.items:
        summary_by_ref[image_result_ref_key(item.ref)] = ImageUnderstandingSummary(
            kind=item.understanding.kind,
            rag_value=item.understanding.rag_value,
            detail_level=item.understanding.detail_level,
            keywords=item.understanding.keywords,
            notes=item.understanding.notes,
            short_description=item.understanding.short_description,
            long_description=item.understanding.long_description,
            model=item.understanding.model,
            thinking=bool(item.understanding.thinking),
            screening_model=item.screening.model,
            screening_thinking=item.screening.thinking,
            status_ok=True,
        )

    return _merge_image_summaries_into_middle(
        middle,
        summary_by_ref,
    )


def _merge_image_summaries_into_middle(
    middle: Middle,
    summary_by_ref: dict[tuple[int, int | None, str], ImageUnderstandingSummary],
) -> Middle:

    merged = middle.model_copy(deep=True)
    for page in merged.pdf_info:
        for block in page.para_blocks:
            _merge_understanding_into_block(
                block,
                page_idx=page.page_idx,
                summary_by_ref=summary_by_ref,
            )
        for block in page.discarded_blocks:
            _merge_understanding_into_block(
                block,
                page_idx=page.page_idx,
                summary_by_ref=summary_by_ref,
            )
        for block in page.preproc_blocks:
            _merge_understanding_into_block(
                block,
                page_idx=page.page_idx,
                summary_by_ref=summary_by_ref,
            )
    return merged


def _merge_understanding_into_block(
    block: Block,
    *,
    page_idx: int,
    summary_by_ref: dict[tuple[int, int | None, str], ImageUnderstandingSummary],
    owner_block_index: int | None = None,
) -> None:
    current_block_index = block.index if block.index is not None else owner_block_index
    for line in block.lines:
        for span in line.spans:
            _merge_understanding_into_span(
                span,
                page_idx=page_idx,
                block_index=current_block_index,
                summary_by_ref=summary_by_ref,
            )
    for child in block.blocks:
        _merge_understanding_into_block(
            child,
            page_idx=page_idx,
            summary_by_ref=summary_by_ref,
            owner_block_index=current_block_index,
        )


def _merge_understanding_into_span(
    span: Span,
    *,
    page_idx: int,
    block_index: int | None,
    summary_by_ref: dict[tuple[int, int | None, str], ImageUnderstandingSummary],
) -> None:
    if span.image_path is None:
        return
    summary = summary_by_ref.get((page_idx, block_index, span.image_path))
    if summary is None:
        return
    span.image_understanding = summary


def iter_understanding_records_from_screening(
    screening_records: Iterable[ScreeningRecord],
    *,
    model: str,
    pricing: ModelPricing,
    base_url: str | None = None,
    api_key: str | None = None,
    thinking: bool = False,
    existing_keys: set[UnderstandingRecordKey] | None = None,
) -> Iterable[UnderstandingRecord]:
    completed_keys = existing_keys if existing_keys is not None else set()
    records = list(screening_records)
    total = len(records)

    for index, screening_record in enumerate(records, start=1):
        screening_run = screening_record.run
        if screening_run is None:
            continue
        completed_key = (
            model,
            thinking,
            screening_record.model,
            screening_record.thinking,
            screening_record_ref_key(screening_record.ref),
        )
        if completed_key in completed_keys:
            logger.info(
                "Skipping existing model=%s thinking=%s source_model=%s source_thinking=%s item=%d/%d image=%s",
                model,
                thinking,
                screening_record.model,
                screening_record.thinking,
                index,
                total,
                screening_record.ref.image_path,
            )
            continue

        logger.info(
            "Processing model=%s thinking=%s source_model=%s source_thinking=%s item=%d/%d image=%s",
            model,
            thinking,
            screening_record.model,
            screening_record.thinking,
            index,
            total,
            screening_record.ref.image_path,
        )
        started = time.perf_counter()
        try:
            run = image_understanding_run_from_screening(
                screening_record.ref,
                screening_run.to_screening_result(),
                model=model,
                base_url=base_url,
                api_key=api_key,
                pricing=pricing,
                thinking=thinking,
            )
            record = UnderstandingRecord(
                ref=screening_record.ref,
                screening=ScreeningSource(
                    model=screening_record.model,
                    thinking=screening_record.thinking,
                    resolved_thinking=screening_record.resolved_thinking,
                    base_url=screening_record.base_url,
                    started_at=screening_record.started_at,
                    latency_sec=screening_record.latency_sec,
                    run=screening_run,
                ),
                model=model,
                thinking=thinking,
                resolved_thinking=thinking,
                base_url=base_url,
                started_at=run.started_at,
                latency_sec=time.perf_counter() - started,
                status=RunStatus(ok=True),
                run=UnderstandingRunView.from_image_understanding_run_result(run),
            )
            logger.info(
                "Completed model=%s thinking=%s source_model=%s source_thinking=%s item=%d/%d image=%s latency=%.2fs",
                model,
                thinking,
                screening_record.model,
                screening_record.thinking,
                index,
                total,
                screening_record.ref.image_path,
                record.latency_sec,
            )
        except (OSError, ValueError, ValidationError) as error:
            record = UnderstandingRecord(
                ref=screening_record.ref,
                screening=ScreeningSource(
                    model=screening_record.model,
                    thinking=screening_record.thinking,
                    resolved_thinking=screening_record.resolved_thinking,
                    base_url=screening_record.base_url,
                    started_at=screening_record.started_at,
                    latency_sec=screening_record.latency_sec,
                    run=screening_run,
                ),
                model=model,
                thinking=thinking,
                resolved_thinking=thinking,
                base_url=base_url,
                started_at=None,
                latency_sec=time.perf_counter() - started,
                status=RunStatus(ok=False, error=str(error)),
                run=None,
            )
            logger.exception(
                "Failed model=%s thinking=%s source_model=%s source_thinking=%s item=%d/%d image=%s",
                model,
                thinking,
                screening_record.model,
                screening_record.thinking,
                index,
                total,
                screening_record.ref.image_path,
            )
        yield record
