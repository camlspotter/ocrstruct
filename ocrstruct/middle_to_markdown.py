from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import Literal, cast

from ocrstruct.middle import Block, Content, ImageUnderstandingSummary, Line, Middle, PageInfo, Result, Span
from ocrstruct.table import MultiCellMode, encode_html_table_eq_tokens, html_tables_to_markdown


type ImageUnderstandingRenderMode = Literal["short", "long"]


@dataclass(frozen=True)
class RenderOptions:
    include_discarded_blocks: bool = False
    include_headers: bool = False
    include_footers: bool = False
    include_page_numbers: bool = False
    include_page_footnotes: bool = True
    include_aside_text: bool = True
    include_images: bool = True
    include_image_understanding: bool = False
    image_understanding_render_mode: ImageUnderstandingRenderMode = "short"
    table_multicell_mode: MultiCellMode | None = None


def source_image_link(image_path: str) -> str:
    rendered_path = _render_image_path(image_path)
    return f'<span class="source-image-link-row"><a href="{rendered_path}" class="source-image-link">👁️‍🗨️</a></span>'


def result_to_markdown(result: Result, *, options: RenderOptions | None = None) -> str:
    return middle_to_markdown(result.middle_json, options=options)


def middle_to_markdown(doc: Middle, *, options: RenderOptions | None = None) -> str:
    opts = options or RenderOptions()
    parts: list[str] = []
    for page in doc.pdf_info:
        page_md = page_to_markdown(page, options=opts)
        if page_md:
            parts.append(page_md)
    return "\n\n".join(parts).strip()


def page_to_markdown(page: PageInfo, *, options: RenderOptions) -> str:
    blocks = page.para_blocks
    if options.include_discarded_blocks:
        blocks = [*blocks, *page.discarded_blocks]

    out: list[str] = []
    for block in blocks:
        block_md = block_to_markdown(block, options=options)
        if block_md:
            out.append(block_md)
    return "\n\n".join(out).strip()


def block_to_markdown(block: Block, *, options: RenderOptions) -> str:
    block_type = block.type

    if block_type in {"header", "header_image"} and not options.include_headers:
        return ""
    if block_type in {"footer", "footer_image"} and not options.include_footers:
        return ""
    if block_type == "page_number" and not options.include_page_numbers:
        return ""
    if block_type == "page_footnote" and not options.include_page_footnotes:
        return ""
    if block_type == "aside_text" and not options.include_aside_text:
        return ""

    if block_type in {"title", "doc_title", "paragraph_title"}:
        title = _block_text(block)
        if not title:
            return ""
        level = block.level
        if level is None:
            if block_type == "doc_title":
                level = 1
            elif block_type == "paragraph_title":
                level = 2
            else:
                level = 1
        level = min(max(level, 1), 6)
        return f"{'#' * level} {title}"

    if block_type in {
        "text",
        "abstract",
        "header",
        "footer",
        "page_number",
        "page_footnote",
        "aside_text",
        "ref_text",
        "phonetic",
        "vertical_text",
        "caption",
        "footnote",
        "image_caption",
        "table_caption",
        "chart_caption",
        "image_footnote",
        "table_footnote",
        "chart_footnote",
        "code_caption",
        "code_footnote",
        "algorithm_caption",
    }:
        return _block_text(block)

    if block_type in {"list", "index"}:
        return _list_block_to_markdown(block, options=options)

    if block_type in {"code", "algorithm", "code_body"}:
        return _code_block_to_markdown(block, options=options)

    if block_type in {"interline_equation", "equation", "formula_number"}:
        return _equation_block_to_markdown(block)

    if block_type in {"image", "image_body", "chart", "chart_body", "seal", "header_image", "footer_image"}:
        return _media_block_to_markdown(block, options=options)

    if block_type in {"table", "table_body"}:
        return _table_block_to_markdown(block, options=options)

    body = _block_text(block)
    child_parts = _child_block_parts(block, options=options)
    return _join_blocks([body, *child_parts])


def line_to_markdown(line: Line) -> str:
    return "".join(span_to_markdown(span) for span in line.spans).strip()


def span_to_markdown(span: Span) -> str:
    content = _content_to_text(span.content)

    if span.type == "inline_equation":
        return _wrap_inline_math(content)
    if span.type == "hyperlink":
        return content
    if span.type == "equation":
        return _wrap_inline_math(content)
    if span.type in {"image", "chart", "seal"} and span.image_path:
        return f"![]({_render_image_path(span.image_path)})"
    if span.type == "table":
        if span.html:
            return span.html
        return content
    if span.type == "interline_equation":
        if content:
            return content
        if span.image_path:
            return f"![]({_render_image_path(span.image_path)})"
        return ""
    return content


def _list_block_to_markdown(block: Block, *, options: RenderOptions) -> str:
    if block.blocks:
        items: list[str] = []
        for child in block.blocks:
            text = _block_text(child)
            if not text:
                text = block_to_markdown(child, options=options).strip()
            if text:
                items.append(f"- {text}")
        if items:
            return "\n".join(items)

    lines = [line_to_markdown(line) for line in block.lines]
    lines = [line for line in lines if line]
    if not lines:
        return ""
    return "\n".join(f"- {line}" for line in lines)


def _code_block_to_markdown(block: Block, *, options: RenderOptions) -> str:
    body = _block_text(_first_child_of_type(block, {"code_body"}) or block)
    if not body:
        body = _block_text(block)
    lang = block.guess_lang or ""
    fenced = f"```{lang}\n{body}\n```".strip()
    image_path = _find_first_image_path(block)
    if image_path:
        fenced = f"{fenced}\n{source_image_link(image_path)}"
    extra = [
        block_to_markdown(child, options=options)
        for child in block.blocks
        if child.type != "code_body"
    ]
    return _join_blocks([fenced, *extra])


def _equation_block_to_markdown(block: Block) -> str:
    text = _block_text(block)
    if text:
        image_path = _find_first_image_path(block)
        out = f"$$\n{text}\n$$"
        if image_path:
            out = f"{out}\n{source_image_link(image_path)}"
        return out

    image_path = _find_first_image_path(block)
    if image_path:
        return f"![equation]({image_path})"
    return ""


def _media_block_to_markdown(block: Block, *, options: RenderOptions) -> str:
    parts: list[str] = []
    image_path = _find_first_image_path(block)
    understanding = _find_first_image_understanding(block)
    if (
        options.include_images
        and image_path is not None
        and options.include_image_understanding
        and understanding is not None
    ):
        parts.append(
            _render_media_with_understanding(
                image_path=image_path,
                understanding=understanding,
                options=options,
            )
        )
    else:
        if options.include_images and image_path:
            parts.append(f"![]({_render_image_path(image_path)})")
        if options.include_image_understanding:
            understanding_md = _render_image_understanding(understanding, options=options)
            if understanding_md:
                parts.append(understanding_md)

    parts.extend(
        block_to_markdown(child, options=options)
        for child in block.blocks
        if child.type not in {"image_body", "chart_body"}
    )
    if not parts:
        text = _block_text(block)
        if text:
            parts.append(text)
    return _join_blocks(parts)


def _table_block_to_markdown(block: Block, *, options: RenderOptions) -> str:
    parts: list[str] = []
    image_path = _find_first_image_path(block)

    html = _find_first_table_html(block)
    if html is None:
        parts.append("<div>警告: HTMLのないテーブル</div>")
    else:
        if options.table_multicell_mode is None:
            table_md = encode_html_table_eq_tokens(html)
        else:
            table_md = html_tables_to_markdown(
                html,
                multicell_mode=options.table_multicell_mode,
            )
        if image_path:
            table_md = f"{table_md}\n{source_image_link(image_path)}"
        parts.append(table_md)

    parts.extend(
        block_to_markdown(child, options=options)
        for child in block.blocks
        if child.type != "table_body"
    )
    return _join_blocks(parts)


def _child_block_parts(block: Block, *, options: RenderOptions) -> list[str]:
    parts: list[str] = []
    for child in block.blocks:
        child_md = block_to_markdown(child, options=options)
        if child_md:
            parts.append(child_md)
    return parts


def _first_child_of_type(block: Block, types: set[str]) -> Block | None:
    for child in block.blocks:
        if child.type in types:
            return child
    return None


def _find_first_image_path(block: Block) -> str | None:
    for line in block.lines:
        for span in line.spans:
            if span.image_path:
                return span.image_path
    for child in block.blocks:
        if image_path := _find_first_image_path(child):
            return image_path
    return None


def _find_first_table_html(block: Block) -> str | None:
    for line in block.lines:
        for span in line.spans:
            if span.html:
                return span.html
    for child in block.blocks:
        if html := _find_first_table_html(child):
            return html
    return None


def _find_first_image_understanding(block: Block) -> ImageUnderstandingSummary | None:
    for line in block.lines:
        for span in line.spans:
            if span.image_understanding is not None:
                return span.image_understanding
    for child in block.blocks:
        if understanding := _find_first_image_understanding(child):
            return understanding
    return None


def _render_image_understanding(
    understanding: ImageUnderstandingSummary | None,
    *,
    options: RenderOptions,
) -> str:
    if understanding is None:
        return ""
    description = _select_image_understanding_description(
        understanding,
        options=options,
    )
    if description is None:
        return ""
    mode_class = f"image-understanding--{options.image_understanding_render_mode}"
    return (
        f'<div class="image-understanding {mode_class}">'
        f"<strong>画像理解:</strong> {escape(description)}"
        "</div>"
    )


def _render_media_with_understanding(
    *,
    image_path: str,
    understanding: ImageUnderstandingSummary,
    options: RenderOptions,
) -> str:
    description = _select_image_understanding_description(
        understanding,
        options=options,
    )
    rendered_path = _render_image_path(image_path)
    media_html = [
        '<div class="image-understanding-layout">',
        '<div class="image-understanding-layout__media">',
        f'<img src="{escape(rendered_path)}" alt="" />',
        source_image_link(image_path),
        "</div>",
    ]
    if description is not None:
        mode_class = f"image-understanding--{options.image_understanding_render_mode}"
        media_html.extend(
            [
                f'<div class="image-understanding {mode_class}">',
                "<strong>画像理解:</strong> "
                f"{escape(description)}",
                "</div>",
            ]
        )
    media_html.append("</div>")
    return "\n".join(media_html)


def _select_image_understanding_description(
    understanding: ImageUnderstandingSummary,
    *,
    options: RenderOptions,
) -> str | None:
    if options.image_understanding_render_mode == "long":
        return understanding.long_description or understanding.short_description
    return understanding.short_description or understanding.long_description


def _block_text(block: Block) -> str:
    lines = [line_to_markdown(line) for line in block.lines]
    lines = [line for line in lines if line]
    return "\n".join(lines).strip()


def _content_to_text(content: Content | None) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(cast(list[str], content))


def _join_blocks(parts: list[str]) -> str:
    return "\n\n".join(part for part in parts if part).strip()


def _wrap_inline_math(text: str) -> str:
    if not text:
        return ""
    return f"${text}$"


def _render_image_path(image_path: str) -> str:
    if "://" in image_path or image_path.startswith("/"):
        return image_path
    if image_path.startswith("images/"):
        return image_path
    return f"images/{image_path}"
