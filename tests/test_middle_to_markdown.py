from __future__ import annotations

from ocrstruct.middle import Block, ImageUnderstandingSummary, Line, Middle, PageInfo, Span
from ocrstruct.middle_to_markdown import RenderOptions, middle_to_markdown


def _middle_with_table(html: str) -> Middle:
    return Middle(
        pdf_info=[
            PageInfo(
                page_idx=0,
                page_size=(100, 100),
                para_blocks=[
                    Block(
                        type="table",
                        lines=[],
                        blocks=[
                            Block(
                                type="table_body",
                                lines=[
                                    Line(
                                        spans=[
                                            Span(
                                                type="table",
                                                html=html,
                                            )
                                        ]
                                    )
                                ],
                            )
                        ],
                    )
                ],
            )
        ]
    )


def test_middle_to_markdown_preserves_default_table_rendering() -> None:
    middle = _middle_with_table(
        "<table><tr><td>A</td><td>B</td></tr></table>"
    )

    rendered = middle_to_markdown(middle)

    assert "<table>" in rendered
    assert "| A | B |" not in rendered


def test_middle_to_markdown_can_expand_multicell_tables_for_llm() -> None:
    middle = _middle_with_table(
        (
            "<table>"
            "<tr><th rowspan='2'>Category</th><th>Q1</th></tr>"
            "<tr><th>Q2</th></tr>"
            "<tr><td>Sales</td><td>10</td></tr>"
            "</table>"
        )
    )

    rendered = middle_to_markdown(
        middle,
        options=RenderOptions(table_multicell_mode="repeat"),
    )

    assert "| Category | Q1 |" in rendered
    assert "| Category | Q2 |" in rendered
    assert "<table>" not in rendered


def test_middle_to_markdown_can_render_image_understanding_summary() -> None:
    middle = Middle(
        pdf_info=[
            PageInfo(
                page_idx=0,
                page_size=(100, 100),
                para_blocks=[
                    Block(
                        type="image",
                        blocks=[
                            Block(
                                type="image_body",
                                lines=[
                                    Line(
                                        spans=[
                                            Span(
                                                type="image",
                                                image_path="sample.png",
                                                image_understanding=ImageUnderstandingSummary(
                                                    kind="diagram",
                                                    rag_value="high",
                                                    detail_level="long",
                                                    keywords=["workflow"],
                                                    short_description="短い説明",
                                                    long_description="詳細な説明です",
                                                    model="understanding-model",
                                                    thinking=False,
                                                    screening_model="screening-model",
                                                    screening_thinking=False,
                                                ),
                                            )
                                        ]
                                    )
                                ],
                            )
                        ],
                    )
                ],
            )
        ]
    )

    rendered = middle_to_markdown(
        middle,
        options=RenderOptions(include_image_understanding=True),
    )

    assert 'class="image-understanding-layout"' in rendered
    assert 'class="image-understanding image-understanding--short"' in rendered
    assert "画像理解:</strong> 短い説明" in rendered


def test_middle_to_markdown_can_render_long_image_understanding_summary() -> None:
    middle = Middle(
        pdf_info=[
            PageInfo(
                page_idx=0,
                page_size=(100, 100),
                para_blocks=[
                    Block(
                        type="image",
                        blocks=[
                            Block(
                                type="image_body",
                                lines=[
                                    Line(
                                        spans=[
                                            Span(
                                                type="image",
                                                image_path="sample.png",
                                                image_understanding=ImageUnderstandingSummary(
                                                    kind="diagram",
                                                    rag_value="high",
                                                    detail_level="long",
                                                    keywords=["workflow"],
                                                    short_description="短い説明",
                                                    long_description="詳細な説明です",
                                                    model="understanding-model",
                                                    thinking=False,
                                                    screening_model="screening-model",
                                                    screening_thinking=False,
                                                ),
                                            )
                                        ]
                                    )
                                ],
                            )
                        ],
                    )
                ],
            )
        ]
    )

    rendered = middle_to_markdown(
        middle,
        options=RenderOptions(
            include_image_understanding=True,
            image_understanding_render_mode="long",
        ),
    )

    assert 'class="image-understanding image-understanding--long"' in rendered
    assert "画像理解:</strong> 詳細な説明です" in rendered
