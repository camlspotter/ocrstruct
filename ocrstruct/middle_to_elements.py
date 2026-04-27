from __future__ import annotations
import logging
import re
import unicodedata

from ocrstruct.types import BBox, Element, Location


logger = logging.getLogger(__name__)


def normalize_text(s: str) -> str:
    out = unicodedata.normalize("NFKC", s)
    out = out.replace("\u3000", " ")
    out = re.sub(r"[ \t]+", " ", out)
    return out.strip()

def _bbox_distance(a: BBox, b: BBox) -> float:
    ax0, ay0, ax1, ay1 = a.as_tuple()
    bx0, by0, bx1, by1 = b.as_tuple()
    dx = max(0.0, max(bx0 - ax1, ax0 - bx1))
    dy = max(0.0, max(by0 - ay1, ay0 - by1))
    return (dx * dx + dy * dy) ** 0.5


def _bbox_union(a: BBox, b: BBox) -> BBox:
    return BBox(
        x0=min(a.x0, b.x0),
        y0=min(a.y0, b.y0),
        x1=max(a.x1, b.x1),
        y1=max(a.y1, b.y1),
    )


def _safe_bbox(block: dict) -> BBox | None:
    bbox = block.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    if not all(isinstance(x, (int, float)) for x in bbox):
        return None
    return BBox(
        x0=float(bbox[0]),
        y0=float(bbox[1]),
        x1=float(bbox[2]),
        y1=float(bbox[3]),
    )


def _safe_loc(block: dict, page_idx: int) -> Location | None:
    if bbox := _safe_bbox(block):
        return Location(bbox= bbox, page_idx= page_idx)
    else:
        return None


def extract_text_from_block(block: dict) -> str:
    lines = block.get("lines")
    if not isinstance(lines, list):
        return ""
    out: list[str] = []
    for line in lines:
        if not isinstance(line, dict):
            continue
        spans = line.get("spans")
        if not isinstance(spans, list):
            continue
        line_text: list[str] = []
        for span in spans:
            if not isinstance(span, dict):
                continue
            content = span.get("content")
            if isinstance(content, str) and content:
                line_text.append(content)
        if line_text:
            out.append("".join(line_text))
    return normalize_text("//".join(out))


def _extract_texts_from_lines(lines: list[dict]) -> list[str]:
    texts: list[str] = []
    for line in lines:
        if not isinstance(line, dict):
            continue
        spans = line.get("spans")
        if not isinstance(spans, list):
            continue
        line_text: list[str] = []
        for span in spans:
            if not isinstance(span, dict):
                continue
            content = span.get("content")
            if isinstance(content, str) and content:
                line_text.append(content)
        if line_text:
            texts.append("".join(line_text))
    return texts


def _extract_text_lines_with_bbox(block: dict, *, page_idx: int) -> list[Element]:
    lines = block.get("lines")
    if not isinstance(lines, list):
        return []

    block_loc = _safe_loc(block, page_idx)
    out: list[Element] = []
    for line in lines:
        if not isinstance(line, dict):
            continue
        spans = line.get("spans")
        if not isinstance(spans, list):
            continue
        line_loc = _safe_loc(line, page_idx) or block_loc
        text_parts: list[str] = []
        text_loc: Location | None = None

        def flush_text() -> None:
            nonlocal text_parts, text_loc
            if not text_parts:
                return
            out.append(Element(
                kind='text',
                text="".join(text_parts),
                loc=text_loc or line_loc,
            ))
            text_parts = []
            text_loc = None

        for span in spans:
            if not isinstance(span, dict):
                continue
            content = span.get("content")
            if not isinstance(content, str) or not content:
                continue
            if bbox := _safe_bbox(span):
                span_loc = Location(bbox=bbox, page_idx=page_idx)
            else:
                span_loc = line_loc
            span_type = span.get("type")
            if span_type == "inline_equation":
                flush_text()
                out.append(Element(
                    kind='math',
                    subkind='inline',
                    text=content,
                    loc=span_loc,
                ))
                continue
            if text_loc is None:
                text_loc = span_loc
            text_parts.append(content)
        flush_text()
    return out


def _warn_missing_bbox(es: list[Element], block_type: str) -> None:
    for e in es:
        if e.kind != 'empty' and e.loc is None:
            logger.warning(
                "Missing bbox for non-empty element (block_type=%s loc=%s): %r",
                block_type,
                e.loc,
                e.to_str(),
            )


def _bbox_sort_key(block: dict) -> tuple[float, float, float, float]:
    bbox = _safe_bbox(block)
    if bbox is None:
        return (1e18, 1e18, 1e18, 1e18)
    return bbox.as_tuple()


def _merge_page_blocks_with_discarded(
    page_idx: int,
    para_blocks: list[dict],
    discarded_blocks: list[dict],
    *,
    header_first_page: dict[str,int],
    footer_first_page: dict[str,int],
) -> list[dict]:
    if not para_blocks:
        return discarded_blocks
    if not discarded_blocks:
        return para_blocks

    n = len(para_blocks)
    slots: list[list[dict]] = [[] for _ in range(n + 1)]

    def _gap_cost(i: int, d: dict) -> float:
        left = para_blocks[0] if i == 0 else para_blocks[i - 1]
        right = para_blocks[-1] if i == n else para_blocks[i]
        db = _safe_bbox(d)
        lb = _safe_bbox(left)
        rb = _safe_bbox(right)
        if db is None or lb is None or rb is None:
            # Fallback: non-geometry items go to tail.
            return float("inf") if i != n else 0.0
        return _bbox_distance(lb, db) + _bbox_distance(rb, db)

    for d in discarded_blocks:
        if d['type'] == 'page_number':
            continue

        text = extract_text_from_block(d)

        definite = False
        ntext = re.sub(r'\s', '', normalize_text(text))
        
        # Ignore empty texts        
        if not ntext:
            continue

        match d['type']:
            case 'header':
                if page_idx == 0:
                    # Headers of the first page are taken, (but not as titles)
                    definite = True
                if re.search(r'.*年.*月.*日', text):
                    # Dates are important
                    definite = True
                if re.search(r'様式|資料|別紙|別表', ntext):
                    definite = True
                if re.search(r'//', text):
                    # Multiline
                    definite = True
                if len(ntext) <= 1:
                    # Too short
                    continue
                if header_first_page.get(text, page_idx) < page_idx:
                    # Duped headers are skipped
                    continue

            case 'footer':
                if re.match(r'<?[0-9]{1,3}>?', ntext):
                    # Page numbers
                    continue
                if re.search(r'allrightsreserved|^copyright', ntext, re.I):
                    # Copyright
                    continue
                if footer_first_page.get(text, page_idx) < page_idx:
                    # Duped footer are skipped
                    continue
                if len(ntext) <= 1:
                    # Too short footer
                    continue
                if re.search(r'//', text):
                    # Multiline
                    definite = True

            case 'page_footnote':
                # Footnotes are important
                definite = True
            
            case 'aside_text':
                if len(ntext) <= 3:
                    # Too short aside texts should be ignored
                    continue
                if re.match(r'([0-9]+//){5}', ntext):
                    # 28//29//30//30//31//33//33//34//36//37//38//41//42//43//44//45//45//46//47//48//49//50
                    continue

        if not definite:
            logger.warning('Salvages %d %s %s', page_idx, d['type'], text)

        best_i = 0
        best_cost = _gap_cost(0, d)
        for i in range(1, n + 1):
            c = _gap_cost(i, d)
            if c < best_cost:
                best_i = i
                best_cost = c
            elif c == best_cost:
                best_is_boundary = best_i in {0, n}
                i_is_interior = 0 < i < n
                if best_is_boundary and i_is_interior:
                    best_i = i
        slots[best_i].append(d)

    for slot in slots:
        slot.sort(key=_bbox_sort_key)

    out: list[dict] = []
    out.extend(slots[0])
    for i, b in enumerate(para_blocks, start=1):
        out.append(b)
        out.extend(slots[i])
    return out


def merge_discarded_blocks_in_middle(middle: dict) -> dict:
    """
    Return middle.json-like dict where each page para_blocks already includes
    discarded_blocks merged by bbox-neighborhood order.
    """
    pdf_info = middle.get("pdf_info")
    if not isinstance(pdf_info, list):
        return middle

    header_first_page, footer_first_page = collect_page_header_footer_texts(middle)

    merged_pages: list[dict] = []
    for page in pdf_info:
        if not isinstance(page, dict):
            continue

        page_idx_raw = page.get("page_idx")
        page_idx = page_idx_raw if isinstance(page_idx_raw, int) else -1
        para_blocks = page.get("para_blocks")
        discarded_blocks = page.get("discarded_blocks")
        if not isinstance(para_blocks, list):
            para_blocks = []
        if not isinstance(discarded_blocks, list):
            discarded_blocks = []

        merged_page = dict(page)
        merged_page["para_blocks"] = _merge_page_blocks_with_discarded(
            page_idx,
            para_blocks,
            discarded_blocks,
            header_first_page= header_first_page,
            footer_first_page= footer_first_page,
        )
        merged_pages.append(merged_page)

    out = dict(middle)
    out["pdf_info"] = merged_pages
    out["_header_text_first_page"] = header_first_page
    out["_footer_text_first_page"] = footer_first_page
    return out


def collect_page_header_footer_texts(
    middle: dict,
) -> tuple[dict[str, int], dict[str, int]]:
    pdf_info = middle.get("pdf_info")
    if not isinstance(pdf_info, list):
        return {}, {}

    header_first_page: dict[str, int] = {}
    footer_first_page: dict[str, int] = {}
    for page in pdf_info:
        if not isinstance(page, dict):
            continue
        page_idx_raw = page.get("page_idx")
        page_idx = page_idx_raw if isinstance(page_idx_raw, int) else -1
        if page_idx == -1:
            continue

        para_blocks = page.get("para_blocks")
        discarded_blocks = page.get("discarded_blocks")
        if not isinstance(para_blocks, list):
            para_blocks = []
        if not isinstance(discarded_blocks, list):
            discarded_blocks = []
        all_blocks = para_blocks + discarded_blocks

        for block in all_blocks:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "header":
                if text := extract_text_from_block(block):
                    t = normalize_text(text)
                    if t and t not in header_first_page:
                        header_first_page[t] = page_idx
            elif block_type == "footer":
                if text := extract_text_from_block(block):
                    t = normalize_text(text)
                    if t and t not in footer_first_page:
                        footer_first_page[t] = page_idx

    return header_first_page, footer_first_page


def _merge_text_from_block(block: dict, *, page_idx: int) -> str:
    out: list[str] = []
    for line in _extract_text_lines_with_bbox(block, page_idx=page_idx):
        out.append(line.to_str())
    return normalize_text("\n".join(out))


def _find_first_image_info(block: dict) -> tuple[str | None, BBox | None]:
    block_bbox = _safe_bbox(block)
    lines = block.get("lines")
    if isinstance(lines, list):
        for line in lines:
            if not isinstance(line, dict):
                continue
            line_bbox = _safe_bbox(line) or block_bbox
            spans = line.get("spans")
            if not isinstance(spans, list):
                continue
            for span in spans:
                if not isinstance(span, dict):
                    continue
                image_path = span.get("image_path")
                if isinstance(image_path, str) and image_path:
                    return image_path, (_safe_bbox(span) or line_bbox)
    children = block.get("blocks")
    if isinstance(children, list):
        for child in children:
            if not isinstance(child, dict):
                continue
            image_path, image_bbox = _find_first_image_info(child)
            if image_path:
                return image_path, image_bbox
    return None, block_bbox


def _join_image_path(img_bucket_path: str, image_path: str) -> str:
    return f"{img_bucket_path.rstrip('/')}/{image_path.lstrip('/')}"


def _block_to_elements(
    block: dict,
    *,
    img_bucket_path: str,
    page_idx: int,
) -> list[Element]:
    block_type = block.get("type")
    if not isinstance(block_type, str):
        return []

    text = _merge_text_from_block(block, page_idx=page_idx)
    text_lines = _extract_text_lines_with_bbox(block, page_idx=page_idx)
    block_loc = _safe_loc(block, page_idx)
    out: list[Element] = []

    if block_type == "title":
        level = block.get("level")
        if isinstance(level, int) and level > 0:
            out.append(Element(
                kind= 'title',
                level= level,
                text= text,
                loc= (text_lines[0].loc or block_loc) if text_lines else block_loc,
            ))
        else:
            if text:
                out.append(Element(
                    kind= 'title',
                    level= None,
                    text=text, 
                    loc= (text_lines[0].loc or block_loc) if text_lines else block_loc,
                ))
        _warn_missing_bbox(out, block_type)
        return out

    if block_type in {"text", "abstract", "header", "footer", "page_number", "page_footnote", "aside_text"}:
        out.extend(text_lines)
        _warn_missing_bbox(out, block_type)
        return out

    if block_type in {"list", "index", "ref_text"}:
        children = block.get("blocks")
        if isinstance(children, list) and children:
            for child in children:
                if not isinstance(child, dict):
                    continue
                child_lines = _extract_text_lines_with_bbox(child, page_idx=page_idx)
                if child_lines:
                    item_text = "\n".join(line.to_str() for line in child_lines).strip()
                    if item_text:
                        out.append(Element(
                            kind= 'item',
                            text= item_text,
                            loc=child_lines[0].loc or block_loc,
                        ))
        elif text:
            out.append(Element(
                kind='item',
                text= text, 
                loc= (text_lines[0].loc or block_loc) if text_lines else block_loc
            ))
        _warn_missing_bbox(out, block_type)
        return out

    if block_type in {"interline_equation"}:
        if text:
            loc= (text_lines[0].loc or block_loc) if text_lines else block_loc
            out.append(Element(kind='math', text=text, loc= loc))
        else:
            image_path, image_bbox = _find_first_image_info(block)
            if image_path:
                out.append(Element(
                    kind='math',
                    image_path= _join_image_path(img_bucket_path, image_path),
                    text='',
                    loc= Location(bbox= image_bbox, page_idx= page_idx) if image_bbox else block_loc
                ))
        _warn_missing_bbox(out, block_type)
        return out

    if block_type in {"image", "table", "chart", "seal"}:
        children = block.get("blocks")
        image_path, image_bbox = _find_first_image_info(block)
        image_loc = Location(bbox= image_bbox, page_idx= page_idx) if image_bbox else None
        caption: list[Element] = []
        footnote: list[Element] = []
        html_body: str | None = None
        html_loc: Location | None = None
        if isinstance(children, list):
            for child in children:
                if not isinstance(child, dict):
                    continue
                ctype = child.get("type")
                if ctype in {"image_caption", "table_caption", "chart_caption"}:
                    caption.extend(
                        line
                        for line in _extract_text_lines_with_bbox(child, page_idx=page_idx)
                        if line.to_str()
                    )
                elif ctype in {"image_footnote", "table_footnote", "chart_footnote"}:
                    footnote.extend(
                        line
                        for line in _extract_text_lines_with_bbox(child, page_idx=page_idx)
                        if line.to_str()
                    )
                elif ctype == "table_body":
                    lines = child.get("lines")
                    if isinstance(lines, list):
                        for line in lines:
                            if not isinstance(line, dict):
                                continue
                            line_loc = _safe_loc(line, page_idx) or _safe_loc(child, page_idx) or block_loc
                            spans = line.get("spans")
                            if not isinstance(spans, list):
                                continue
                            for span in spans:
                                if not isinstance(span, dict):
                                    continue
                                html = span.get("html")
                                if isinstance(html, str) and html:
                                    html_body = html
                                    html_loc = _safe_loc(span, page_idx) or line_loc
                                    break
                            if html_body:
                                break
                    if html_body:
                        break
        if caption:
            out.extend(caption)

        if image_path:
            match block_type:
                case 'image':
                    out.append(Element(
                        kind='image',
                        image_path= _join_image_path(img_bucket_path, image_path),
                        text= None,
                        loc= image_loc,
                    ))
                case 'table':
                    out.append(Element(
                        kind='table',
                        image_path= _join_image_path(img_bucket_path, image_path),
                        text= html_body,
                        loc= image_loc
                    ))
                case 'chart':
                    out.append(Element(
                        kind='chart',
                        image_path= _join_image_path(img_bucket_path, image_path),
                        text= None,
                        loc= image_loc,
                    ))
                case 'seal':                
                    out.append(Element(
                        kind='seal',
                        image_path= _join_image_path(img_bucket_path, image_path),
                        text= None,
                        loc= image_loc,
                    ))
        else:
            assert html_body is None
        if footnote:
            out.extend(footnote)
        _warn_missing_bbox(out, block_type)
        return out

    if block_type == "code":
        lang = block.get("guess_lang")
        if not isinstance(lang, str) or not lang:
            lang = ""
        body = ""
        children = block.get("blocks")
        if isinstance(children, list):
            for child in children:
                if isinstance(child, dict) and child.get("type") == "code_body":
                    body = _merge_text_from_block(child, page_idx=page_idx)
                    break
        if not body:
            body = text
        code_loc = (text_lines[0].loc or block_loc) if text_lines else block_loc
        out.append(Element(kind='code', subkind= lang, text=body, loc= code_loc))
        _warn_missing_bbox(out, block_type)
        return out

    raise RuntimeError(f"Unsupported block_type: {block_type}")


def to_elements(middle: dict, *, img_bucket_path: str) -> list[Element]:
    prepared_middle = merge_discarded_blocks_in_middle(middle)
    pdf_info = prepared_middle.get("pdf_info")
    if not isinstance(pdf_info, list):
        raise ValueError("middle.json does not have valid 'pdf_info'")

    out: list[Element] = []
    for page in pdf_info:
        if not isinstance(page, dict):
            continue
        page_idx_raw = page.get("page_idx")
        page_idx = page_idx_raw if isinstance(page_idx_raw, int) else -1
        para_blocks = page.get("para_blocks")
        if not isinstance(para_blocks, list):
            continue
        for block in para_blocks:
            if not isinstance(block, dict):
                continue
            out.extend(
                _block_to_elements(
                    block,
                    img_bucket_path=img_bucket_path,
                    page_idx=page_idx,
                )
            )
            if out and out[-1].kind != 'empty':
                out.append(Element(kind='empty', text=None, loc=None))
    while out and out[-1].kind == 'empty':
        out.pop()
    return out


def to_markdown(middle: dict, *, img_bucket_path: str) -> str:
    es = to_elements(middle, img_bucket_path=img_bucket_path)
    return "\n".join(e.to_str() for e in es)
