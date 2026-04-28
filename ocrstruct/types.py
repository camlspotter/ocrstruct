from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from ocrstruct.table import encode_html_table_eq_tokens, html_tables_to_markdown


class BBox(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)


class Location(BaseModel):
    bbox: BBox
    page_idx: int


class LinkRegion(BaseModel):
    page_idx: int
    bbox: BBox
    target_kind: Literal["external", "internal", "unknown"]
    uri: str | None = None
    dest_page_idx: int | None = None
    dest_raw: str | None = None


class Element(BaseModel):
    kind: Literal[
        "empty",
        "text",
        "item",
        "image",
        "title",
        "math",
        "code",
        "table",
        "chart",
        "seal",
    ]
    subkind: str | None = None
    image_path: str | None = None
    alt : str | None = None
    description: str | None = None
    level: int | None = None
    text: str | None
    loc: Location | None

    def to_markdown(self) -> str:
        '''For human-eyes'''
        match self.kind:
            case "empty":
                assert self.text is None
                assert self.subkind is None
                assert self.image_path is None
                assert self.alt is None
                assert self.description is None
                assert self.level is None
                return ""
            case "text":
                assert self.text is not None
                assert self.subkind is None
                assert self.image_path is None
                assert self.alt is None
                assert self.description is None
                assert self.level is None
                return self.text
            case "item":
                assert self.text is not None
                assert self.subkind is None
                assert self.image_path is None
                assert self.alt is None
                assert self.description is None
                assert self.level is None
                return f"- {self.text}"
            case "image":
                assert self.image_path
                assert self.text is None
                assert self.subkind is None
                assert self.level is None
                alt = self.alt or ''
                desc = f'(画像説明: {self.description})' if self.description else ''
                return f"![{alt}]({self.image_path}){desc}"
            case "title":
                assert self.text is not None
                assert self.subkind is None
                assert self.image_path is None
                assert self.alt is None
                assert self.description is None
                if self.level:
                    return f"{'#' * self.level} {self.text}"
                return self.text
            case "math":
                assert self.level is None
                if self.text:
                    img_link = self.source_image_link() if self.image_path else ''
                    if self.subkind == "inline":
                        return f"${self.text}${img_link}"
                    else:
                        return f"$$\n{self.text}\n$${img_link}"
                alt = self.alt or '数式'
                desc = f'(数式画像説明: {self.description})' if self.description else ''
                return f"![{alt}]({self.image_path})"
            case "code":
                assert self.text is not None
                assert self.image_path is None
                assert self.alt is None
                assert self.description is None
                assert self.level is None
                return f"```{self.subkind or ''}\n{self.text}```"
            case "table":
                assert self.text is not None
                assert self.subkind is None
                assert self.level is None
                text = encode_html_table_eq_tokens(self.text)
                if self.image_path:
                    return f'{text}\n{self.source_image_link()}'
                else:
                    return text
            case "chart":
                assert self.text is None
                assert self.subkind is None
                assert self.image_path is not None
                assert self.level is None
                return f"![チャート画像]({self.image_path})"
            case "seal":
                assert self.text is None
                assert self.subkind is None
                assert self.image_path is not None
                assert self.level is None
                return f"![印章画像]({self.image_path})"

    def to_markdown_for_payload(self) -> str:
        '''For LLM'''
        match self.kind:
            case "empty":
                assert self.text is None
                return ""
            case "text":
                assert self.text is not None
                return self.text
            case "item":
                assert self.text is not None
                return f"- {self.text}"
            case "image":
                assert self.image_path
                assert self.text is None
                alt = self.alt or ''
                desc = f'(画像説明: {self.description})' if self.description else ''
                return f"![{alt}]({self.image_path}){desc}"
            case "title":
                assert self.text is not None
                if self.level:
                    return f"{'#' * self.level} {self.text}"
                return self.text
            case "math":
                if self.text:
                    if self.subkind == "inline":
                        return f"${self.text}$"
                    return f"$$\n{self.text}\n$$"
                alt = self.alt or '数式'
                desc = f'(数式画像説明: {self.description})' if self.description else ''
                return f"![{alt}]({self.image_path}){desc}"
            case "code":
                assert self.text is not None
                return f"```{self.subkind or ''}\n{self.text}```"
            case "table":
                assert self.text is not None
                text = html_tables_to_markdown(self.text)
                if self.image_path:
                    return f"{text} (テーブル画像: ![]({self.image_path}))"
                else:
                    return text
            case "chart":
                assert self.text is None
                return f"![チャート画像]({self.image_path})"
            case "seal":
                assert self.text is None
                return f"![印章画像]({self.image_path})"

    def to_str(self) -> str:
        return self.to_markdown_for_payload()

    def embed_alt_and_description(self) -> str:
        match self.alt, self.description:
            case None, None:
                return ''
            case None, _:
                return '画像({self.description})'
            case _, None:
                return '画像({self.alt})'
            case _:
                return '画像({self.alt} {self.description})'

    def to_markdown_for_embedding(self) -> str:
        '''
        - Drop images without alt nor description.
        - Huge tables are compacted
        '''
        match self.kind:
            case "empty":
                assert self.text is None
                return ""
            case "text":
                assert self.text is not None
                return self.text
            case "item":
                assert self.text is not None
                return f"- {self.text}"
            case "image":
                assert self.image_path
                assert self.text is None
                if text := self.embed_alt_and_description():
                    return f'画像({text})'
                else:
                    return ''
            case "title":
                assert self.text is not None
                if self.level:
                    return f"{'#' * self.level} {self.text}"
                return self.text
            case "math":
                if self.text:
                    if self.subkind == "inline":
                        return f"${self.text}$"
                    return f"$$\n{self.text}\n$$"
                if text := self.embed_alt_and_description():
                    return f'数式({text})'
                else:
                    return ''
            case "code":
                assert self.text is not None
                return f"```{self.subkind or ''}\n{self.text}```"
            case "table":
                assert self.text is not None
                text = html_tables_to_markdown(self.text)
                return text
            case "chart":
                assert self.text is None
                if text := self.embed_alt_and_description():
                    return f'チャート({text})'
                else:
                    return ''
            case "seal":
                assert self.text is None
                if text := self.embed_alt_and_description():
                    return f"印章画像({text})"
                else:
                    return ''

    def source_image_link(self):
        assert self.image_path
        return f'<span class="source-image-link-row"><a href="{self.image_path}" class="source-image-link">👁️‍🗨️</a></span>'
