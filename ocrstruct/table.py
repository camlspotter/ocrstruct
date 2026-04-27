from __future__ import annotations

import base64
import re
from typing import Literal
from bs4 import BeautifulSoup, Tag


_TABLE_RE = re.compile(r"<table\b[^>]*>.*?</table>", re.IGNORECASE | re.DOTALL)
_EQ_RE = re.compile(r"<eq>(.*?)</eq>", re.IGNORECASE | re.DOTALL)
_EQ_TOKEN_RE = re.compile(r"CODEXEQ\[([A-Za-z0-9_\-=]+)\]")
type MultiCellMode = Literal["blank", "repeat", "keep_html"]


def _normalize_cell_text(text: str) -> str:
    s = re.sub(r"\s+", " ", text).strip()
    # Keep markdown table structure safe.
    s = s.replace("|", r"\|")
    return s


def _parse_span(value: object | None) -> int:
    if value is None:
        return 1
    try:
        if isinstance(value, (list, tuple)):
            if not value:
                return 1
            value = value[0]
        n = int(str(value))
        return n if n > 0 else 1
    except Exception:
        return 1


def _table_tag_to_markdown(table: Tag, multicell_mode: MultiCellMode) -> str | None:
    trs = table.find_all("tr")
    if not trs:
        return None

    # Expanded grid with rowspan/colspan support.
    grid: list[list[str]] = []
    rowspan_left: list[int] = []
    rowspan_text: list[str] = []

    max_cols = 0
    has_header = False

    for tr in trs:
        cells = tr.find_all(["th", "td"], recursive=False)
        if not cells:
            continue

        row: list[str] = []
        cidx = 0

        def fill_rowspan_columns() -> None:
            nonlocal cidx
            while cidx < len(rowspan_left) and rowspan_left[cidx] > 0:
                row.append(rowspan_text[cidx])
                rowspan_left[cidx] -= 1
                cidx += 1

        fill_rowspan_columns()
        for cell in cells:
            while cidx < len(rowspan_left) and rowspan_left[cidx] > 0:
                row.append(rowspan_text[cidx])
                rowspan_left[cidx] -= 1
                cidx += 1

            txt = _normalize_cell_text(cell.get_text(" ", strip=True))
            rowspan = _parse_span(cell.get("rowspan"))
            colspan = _parse_span(cell.get("colspan"))
            has_multicell = rowspan > 1 or colspan > 1
            if has_multicell and multicell_mode == "keep_html":
                return None

            for j in range(colspan):
                if multicell_mode == "blank" and has_multicell and j > 0:
                    cell_txt = ""
                else:
                    cell_txt = txt
                row.append(cell_txt)
                if cidx >= len(rowspan_left):
                    rowspan_left.append(0)
                    rowspan_text.append("")
                if rowspan > 1:
                    rowspan_left[cidx] = max(rowspan_left[cidx], rowspan - 1)
                    # Expanded area in following rows.
                    if multicell_mode == "blank":
                        rowspan_text[cidx] = ""
                    else:
                        rowspan_text[cidx] = cell_txt
                cidx += 1

            if cell.name == "th":
                has_header = True

        fill_rowspan_columns()
        max_cols = max(max_cols, len(row))
        grid.append(row)

    if not grid:
        return None

    for row in grid:
        if len(row) < max_cols:
            row.extend([""] * (max_cols - len(row)))

    header_idx = 0 if has_header else None
    if header_idx is None:
        header = [f"col{i+1}" for i in range(max_cols)]
        body = grid
    else:
        header = grid[header_idx]
        body = grid[header_idx + 1 :]

    def mk_line(cols: list[str]) -> str:
        return "| " + " | ".join(cols) + " |"

    sep = "| " + " | ".join(["---"] * max_cols) + " |"
    lines = [mk_line(header), sep]
    lines.extend(mk_line(r) for r in body)
    return "\n".join(lines)


def html_tables_to_markdown(text: str, multicell_mode: MultiCellMode = "repeat") -> str:
    """
    Convert inline HTML <table> blocks in markdown text to markdown tables when possible.
    multicell_mode:
      - "blank": expand rowspan/colspan, but expanded cells are empty.
      - "repeat": expand rowspan/colspan with repeated values.
      - "keep_html": if any rowspan/colspan exists, keep original HTML table.
    """
    if "<table" not in text.lower():
        return text

    def repl(m: re.Match[str]) -> str:
        table_html = m.group(0)
        soup = BeautifulSoup(table_html, "html.parser")
        table = soup.find("table")
        if table is None:
            return table_html
        md = _table_tag_to_markdown(table, multicell_mode=multicell_mode)
        if md is None:
            return table_html
        return "\n" + md + "\n"

    return _TABLE_RE.sub(repl, text)


def html_table_eq_to_mathjax(text: str) -> str:
    if "<eq>" not in text.lower():
        return text

    def repl(m: re.Match[str]) -> str:
        expr = m.group(1).strip()
        return f'<span class="math inline">\\({expr}\\)</span>'

    return _EQ_RE.sub(repl, text)


def encode_html_table_eq_tokens(text: str) -> str:
    if "<eq>" not in text.lower():
        return text

    def repl(m: re.Match[str]) -> str:
        expr = m.group(1).strip()
        encoded = base64.urlsafe_b64encode(expr.encode("utf-8")).decode("ascii")
        return f"CODEXEQ[{encoded}]"

    return _EQ_RE.sub(repl, text)


def decode_html_table_eq_tokens(text: str) -> str:
    if "CODEXEQ[" not in text:
        return text

    def repl(m: re.Match[str]) -> str:
        encoded = m.group(1)
        expr = base64.urlsafe_b64decode(encoded.encode("ascii")).decode("utf-8")
        return f'<span class="math inline">\\({expr}\\)</span>'

    return _EQ_TOKEN_RE.sub(repl, text)
