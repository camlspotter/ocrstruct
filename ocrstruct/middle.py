from __future__ import annotations

import logging
import re
import unicodedata
from typing import Literal, TypeAlias

from pydantic import ConfigDict, Field

from ocrstruct.utils import BaseModelWithSave


type BBox = tuple[float, float, float, float]
type PageSize = tuple[int, int]
type Content = str | list[str]


logger = logging.getLogger(__name__)


class Model(BaseModelWithSave):
    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
        serialize_by_alias=True,
    )


SpanType: TypeAlias = Literal[
    "text",
    "image",
    "table",
    "chart",
    "interline_equation",
    "inline_equation",
    "equation",
    "hyperlink",
    "seal",
]


class Span(Model):
    type: str # SpanType but may contain an unexpected
    bbox: BBox | None = None
    content: Content | None = None
    score: float | None = None
    image_path: str | None = None
    html: str | None = None


class Line(Model):
    bbox: BBox | None = None
    spans: list[Span] = Field(default_factory=list)


BlockType: TypeAlias = Literal[
    "image",
    "table",
    "chart",
    "image_body",
    "table_body",
    "chart_body",
    "caption",
    "image_caption",
    "table_caption",
    "chart_caption",
    "algorithm_caption",
    "footnote",
    "image_footnote",
    "table_footnote",
    "chart_footnote",
    "text",
    "title",
    "interline_equation",
    "equation",
    "list",
    "index",
    "discarded",
    "code",
    "code_body",
    "code_caption",
    "code_footnote",
    "algorithm",
    "ref_text",
    "phonetic",
    "header",
    "footer",
    "page_number",
    "aside_text",
    "page_footnote",
    "abstract",
    "doc_title",
    "paragraph_title",
    "vertical_text",
    "seal",
    "header_image",
    "footer_image",
    "formula_number",
]


class Block(Model):
    type: str # BlockType but may contain an unexpected
    bbox: BBox | None = None
    lines: list[Line] = Field(default_factory=list)
    blocks: list["Block"] = Field(default_factory=list)
    index: int | None = None
    level: int | None = None
    guess_lang: str | None = None
    line_avg_height: int | None = None
    bbox_fs: BBox | None = None
    page_num: int | None = None
    page_size: PageSize | None = None


class PageInfo(Model):
    page_idx: int
    page_size: PageSize
    para_blocks: list[Block] = Field(default_factory=list)
    discarded_blocks: list[Block] = Field(default_factory=list)
    preproc_blocks: list[Block] = Field(default_factory=list)


class Middle(Model):
    pdf_info: list[PageInfo]
    backend: str | None = Field(default=None, alias="_backend")
    version_name: str | None = Field(default=None, alias="_version_name")
    ocr_enable: bool | None = Field(default=None, alias="_ocr_enable")
    vlm_ocr_enable: bool | None = Field(default=None, alias="_vlm_ocr_enable")
    header_text_first_page: dict[str, int] | None = Field(default=None, alias="_header_text_first_page")
    footer_text_first_page: dict[str, int] | None = Field(default=None, alias="_footer_text_first_page")


# middle.json has this type
class Result(Model):
    middle_json: Middle
    extracted_by: str


def normalize_text(s: str) -> str:
    out = unicodedata.normalize("NFKC", s)
    out = out.replace("\u3000", " ")
    out = re.sub(r"[ \t]+", " ", out)
    return out.strip()


def _content_to_text(content: Content | None) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(content)


def extract_text_from_block(block: Block) -> str:
    out: list[str] = []
    for line in block.lines:
        line_text: list[str] = []
        for span in line.spans:
            content = _content_to_text(span.content)
            if content:
                line_text.append(content)
        if line_text:
            out.append("".join(line_text))
    return normalize_text("//".join(out))


def _bbox_distance(a: BBox, b: BBox) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    dx = max(0.0, max(bx0 - ax1, ax0 - bx1))
    dy = max(0.0, max(by0 - ay1, ay0 - by1))
    return (dx * dx + dy * dy) ** 0.5


def _bbox_sort_key(block: Block) -> tuple[float, float, float, float]:
    if block.bbox is None:
        return (1e18, 1e18, 1e18, 1e18)
    return block.bbox


def _merge_page_blocks_with_discarded(
    page_idx: int,
    para_blocks: list[Block],
    discarded_blocks: list[Block],
    *,
    header_first_page: dict[str, int],
    footer_first_page: dict[str, int],
) -> list[Block]:
    if not para_blocks:
        return discarded_blocks
    if not discarded_blocks:
        return para_blocks

    n = len(para_blocks)
    slots: list[list[Block]] = [[] for _ in range(n + 1)]

    def _gap_cost(i: int, discarded: Block) -> float:
        left = para_blocks[0] if i == 0 else para_blocks[i - 1]
        right = para_blocks[-1] if i == n else para_blocks[i]
        if discarded.bbox is None or left.bbox is None or right.bbox is None:
            return float("inf") if i != n else 0.0
        return _bbox_distance(left.bbox, discarded.bbox) + _bbox_distance(right.bbox, discarded.bbox)

    for discarded in discarded_blocks:
        if discarded.type == "page_number":
            continue

        text = extract_text_from_block(discarded)
        definite = False
        ntext = re.sub(r"\s", "", normalize_text(text))
        if not ntext:
            continue

        match discarded.type:
            case "header":
                if page_idx == 0:
                    definite = True
                if re.search(r".*年.*月.*日", text):
                    definite = True
                if re.search(r"様式|資料|別紙|別表", ntext):
                    definite = True
                if re.search(r"//", text):
                    definite = True
                if len(ntext) <= 1:
                    continue
                if header_first_page.get(text, page_idx) < page_idx:
                    continue
            case "footer":
                if re.match(r"<?[0-9]{1,3}>?", ntext):
                    continue
                if re.search(r"allrightsreserved|^copyright", ntext, re.I):
                    continue
                if footer_first_page.get(text, page_idx) < page_idx:
                    continue
                if len(ntext) <= 1:
                    continue
                if re.search(r"//", text):
                    definite = True
            case "page_footnote":
                definite = True
            case "aside_text":
                if len(ntext) <= 3:
                    continue
                if re.match(r"([0-9]+//){5}", ntext):
                    continue

        if not definite:
            logger.warning("Salvages %d %s %s", page_idx, discarded.type, text)

        best_i = 0
        best_cost = _gap_cost(0, discarded)
        for i in range(1, n + 1):
            cost = _gap_cost(i, discarded)
            if cost < best_cost:
                best_i = i
                best_cost = cost
            elif cost == best_cost:
                best_is_boundary = best_i in {0, n}
                i_is_interior = 0 < i < n
                if best_is_boundary and i_is_interior:
                    best_i = i
        slots[best_i].append(discarded)

    for slot in slots:
        slot.sort(key=_bbox_sort_key)

    out: list[Block] = []
    out.extend(slots[0])
    for i, block in enumerate(para_blocks, start=1):
        out.append(block)
        out.extend(slots[i])
    return out


def collect_page_header_footer_texts(middle: Middle) -> tuple[dict[str, int], dict[str, int]]:
    header_first_page: dict[str, int] = {}
    footer_first_page: dict[str, int] = {}

    for page in middle.pdf_info:
        page_idx = page.page_idx
        if page_idx == -1:
            continue

        for block in [*page.para_blocks, *page.discarded_blocks]:
            if block.type == "header":
                text = extract_text_from_block(block)
                if text:
                    normalized = normalize_text(text)
                    if normalized and normalized not in header_first_page:
                        header_first_page[normalized] = page_idx
            elif block.type == "footer":
                text = extract_text_from_block(block)
                if text:
                    normalized = normalize_text(text)
                    if normalized and normalized not in footer_first_page:
                        footer_first_page[normalized] = page_idx

    return header_first_page, footer_first_page


def merge_discarded_blocks(middle: Middle) -> Middle:
    header_first_page, footer_first_page = collect_page_header_footer_texts(middle)

    merged_pages: list[PageInfo] = []
    for page in middle.pdf_info:
        merged_page = page.model_copy(deep=True)
        merged_page.para_blocks = _merge_page_blocks_with_discarded(
            page.page_idx,
            merged_page.para_blocks,
            merged_page.discarded_blocks,
            header_first_page=header_first_page,
            footer_first_page=footer_first_page,
        )
        merged_pages.append(merged_page)

    out = middle.model_copy(deep=True)
    out.pdf_info = merged_pages
    out.header_text_first_page = header_first_page
    out.footer_text_first_page = footer_first_page
    return out
