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

    assert "<table>" not in rendered
    assert "| A | B |" in rendered


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


def test_middle_to_markdown_does_not_invent_column_names_for_headerless_tables() -> None:
    middle = _middle_with_table("<table><tr><td>A</td><td>B</td></tr></table>")

    rendered = middle_to_markdown(
        middle,
        options=RenderOptions(table_multicell_mode="repeat"),
    )

    assert "| col1 |" not in rendered
    assert "| col2 |" not in rendered
    assert "|  |  |" in rendered
    assert "| A | B |" in rendered


def test_middle_to_markdown_keeps_inline_math_markers_in_markdown_tables() -> None:
    middle = _middle_with_table(
        "<table><tr><th>col</th></tr><tr><td><eq>\\alpha + \\beta</eq></td></tr></table>"
    )

    rendered = middle_to_markdown(
        middle,
        options=RenderOptions(table_multicell_mode="repeat"),
    )

    assert "| col |" in rendered
    assert "| $\\alpha+ \\beta$ |" in rendered


def test_middle_to_markdown_can_render_table_math_as_unicode_text() -> None:
    middle = _middle_with_table(
        "<table><tr><th>col</th></tr><tr><td><eq>\\alpha + \\beta</eq></td></tr></table>"
    )

    rendered = middle_to_markdown(
        middle,
        options=RenderOptions(
            table_multicell_mode="repeat",
            render_latex_as_unicode_text=True,
        ),
    )

    assert "| col |" in rendered
    assert "| $α+ β$ |" in rendered


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
        options=RenderOptions(include_image_understanding="html"),
    )

    assert 'class="image-understanding image-understanding--long"' in rendered
    assert "diagram 画像:</strong> 詳細な説明です" in rendered


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
            include_image_understanding="html",
            image_understanding_render_mode="long",
        ),
    )

    assert 'class="image-understanding image-understanding--long"' in rendered
    assert "diagram 画像:</strong> 詳細な説明です" in rendered


def test_middle_to_markdown_includes_source_image_links_by_default() -> None:
    middle = Middle(
        pdf_info=[
            PageInfo(
                page_idx=0,
                page_size=(100, 100),
                para_blocks=[
                    Block(
                        type="table",
                        blocks=[
                            Block(
                                type="table_body",
                                lines=[
                                    Line(
                                        spans=[
                                            Span(
                                                type="table",
                                                html="<table><tr><td>A</td></tr></table>",
                                                image_path="table.png",
                                            )
                                        ]
                                    )
                                ],
                            )
                        ],
                    ),
                    Block(
                        type="equation",
                        lines=[
                            Line(
                                spans=[
                                    Span(
                                        type="equation",
                                        content="x + y = z",
                                        image_path="equation.png",
                                    )
                                ]
                            )
                        ],
                    ),
                    Block(
                        type="code",
                        guess_lang="python",
                        lines=[
                            Line(
                                spans=[
                                    Span(
                                        type="text",
                                        content="print('hello')",
                                        image_path="code.png",
                                    )
                                ]
                            )
                        ],
                    ),
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
                    ),
                ],
            )
        ]
    )

    rendered = middle_to_markdown(
        middle,
        options=RenderOptions(include_image_understanding="html"),
    )

    assert rendered.count('class="source-image-link"') == 3
    assert "![](images/sample.png)" in rendered


def test_middle_to_markdown_can_disable_source_image_links() -> None:
    middle = Middle(
        pdf_info=[
            PageInfo(
                page_idx=0,
                page_size=(100, 100),
                para_blocks=[
                    Block(
                        type="table",
                        blocks=[
                            Block(
                                type="table_body",
                                lines=[
                                    Line(
                                        spans=[
                                            Span(
                                                type="table",
                                                html="<table><tr><td>A</td></tr></table>",
                                                image_path="table.png",
                                            )
                                        ]
                                    )
                                ],
                            )
                        ],
                    ),
                    Block(
                        type="equation",
                        lines=[
                            Line(
                                spans=[
                                    Span(
                                        type="equation",
                                        content="x + y = z",
                                        image_path="equation.png",
                                    )
                                ]
                            )
                        ],
                    ),
                    Block(
                        type="code",
                        guess_lang="python",
                        lines=[
                            Line(
                                spans=[
                                    Span(
                                        type="text",
                                        content="print('hello')",
                                        image_path="code.png",
                                    )
                                ]
                            )
                        ],
                    ),
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
                    ),
                ],
            )
        ]
    )

    rendered = middle_to_markdown(
        middle,
        options=RenderOptions(
            include_image_understanding="html",
            include_source_image_links=False,
        ),
    )

    assert 'class="source-image-link"' not in rendered


def test_middle_to_markdown_can_render_rag_image_understanding_summary() -> None:
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
        options=RenderOptions(include_image_understanding="rag"),
    )

    assert "<image_summary>" in rendered
    assert "kind=diagram" in rendered
    assert 'keywords="workflow"' in rendered
    assert "description: 詳細な説明です" in rendered
    assert 'class="image-understanding-layout"' not in rendered


def test_middle_to_markdown_can_render_latex_as_unicode_text() -> None:
    middle = Middle(
        pdf_info=[
            PageInfo(
                page_idx=0,
                page_size=(100, 100),
                para_blocks=[
                    Block(
                        type="text",
                        lines=[
                            Line(
                                spans=[
                                    Span(type="text", content="Inline "),
                                    Span(type="inline_equation", content=r"\alpha_i^2 + \beta"),
                                    Span(type="text", content=" done"),
                                ]
                            )
                        ],
                    ),
                    Block(
                        type="equation",
                        lines=[
                            Line(
                                spans=[
                                    Span(
                                        type="equation",
                                        content=r"\frac{\alpha_1 + \beta_2}{\gamma^2}",
                                    )
                                ]
                            )
                        ],
                    ),
                ],
            )
        ]
    )

    rendered = middle_to_markdown(
        middle,
        options=RenderOptions(render_latex_as_unicode_text=True),
    )

    assert "Inline $α_i^2 + β$ done" in rendered
    assert "$$\n\\frac{α_1 + β_2}{γ^2}\n$$" in rendered
    assert r"\alpha" not in rendered


def test_middle_to_markdown_normalizes_math_spacing_before_rendering() -> None:
    middle = Middle(
        pdf_info=[
            PageInfo(
                page_idx=0,
                page_size=(100, 100),
                para_blocks=[
                    Block(
                        type="text",
                        lines=[
                            Line(
                                spans=[
                                    Span(
                                        type="inline_equation",
                                        content=r" \alpha _ { i } ^ { 2 } + \beta ",
                                    )
                                ]
                            )
                        ],
                    ),
                    Block(
                        type="equation",
                        lines=[
                            Line(
                                spans=[
                                    Span(
                                        type="equation",
                                        content=r" \frac { \alpha _ 1 + \beta _ 2 } { \gamma ^ 2 } ",
                                    )
                                ]
                            )
                        ],
                    ),
                ],
            )
        ]
    )

    rendered = middle_to_markdown(
        middle,
        options=RenderOptions(render_latex_as_unicode_text=True),
    )

    assert "$α_i^2+ β$" in rendered
    assert "$$\n\\frac{α_1 + β_2}{γ^2}\n$$" in rendered


def test_middle_to_markdown_collapses_spaced_subscript_letters() -> None:
    middle = Middle(
        pdf_info=[
            PageInfo(
                page_idx=0,
                page_size=(100, 100),
                para_blocks=[
                    Block(
                        type="text",
                        lines=[
                            Line(
                                spans=[
                                    Span(
                                        type="inline_equation",
                                        content=r"\alpha_{p r o b}",
                                    )
                                ]
                            )
                        ],
                    )
                ],
            )
        ]
    )

    rendered = middle_to_markdown(
        middle,
        options=RenderOptions(render_latex_as_unicode_text=True),
    )

    assert "$α_{prob}$" in rendered


def test_middle_to_markdown_collapses_spaced_letters_inside_mathrm_subscript() -> None:
    middle = Middle(
        pdf_info=[
            PageInfo(
                page_idx=0,
                page_size=(100, 100),
                para_blocks=[
                    Block(
                        type="text",
                        lines=[
                            Line(
                                spans=[
                                    Span(
                                        type="inline_equation",
                                        content=r"\alpha_{\mathrm{p r o b}}",
                                    )
                                ]
                            )
                        ],
                    )
                ],
            )
        ]
    )

    rendered = middle_to_markdown(
        middle,
        options=RenderOptions(render_latex_as_unicode_text=True),
    )

    assert "$α_{prob}$" in rendered


def test_middle_to_markdown_handles_malformed_mathrm_subscript() -> None:
    middle = Middle(
        pdf_info=[
            PageInfo(
                page_idx=0,
                page_size=(100, 100),
                para_blocks=[
                    Block(
                        type="text",
                        lines=[
                            Line(
                                spans=[
                                    Span(
                                        type="inline_equation",
                                        content=r"\alpha_\mathrm(prob}",
                                    )
                                ]
                            )
                        ],
                    )
                ],
            )
        ]
    )

    rendered = middle_to_markdown(
        middle,
        options=RenderOptions(render_latex_as_unicode_text=True),
    )

    assert "$α_\\mathrm(prob}$" in rendered


def test_middle_to_markdown_unwraps_mathit_and_mathbb() -> None:
    middle = Middle(
        pdf_info=[
            PageInfo(
                page_idx=0,
                page_size=(100, 100),
                para_blocks=[
                    Block(
                        type="text",
                        lines=[
                            Line(
                                spans=[
                                    Span(
                                        type="inline_equation",
                                        content=r"\mathit{prob} + \mathbb{R} + \alpha",
                                    )
                                ]
                            )
                        ],
                    )
                ],
            )
        ]
    )

    rendered = middle_to_markdown(
        middle,
        options=RenderOptions(render_latex_as_unicode_text=True),
    )

    assert "${prob}+ R+ α$" in rendered


def test_middle_to_markdown_drops_braces_for_single_char_subscripts_and_superscripts() -> None:
    middle = Middle(
        pdf_info=[
            PageInfo(
                page_idx=0,
                page_size=(100, 100),
                para_blocks=[
                    Block(
                        type="text",
                        lines=[
                            Line(
                                spans=[
                                    Span(
                                        type="inline_equation",
                                        content=r"e_{x} + f^{y} + g_{prob}",
                                    )
                                ]
                            )
                        ],
                    )
                ],
            )
        ]
    )

    rendered = middle_to_markdown(
        middle,
        options=RenderOptions(render_latex_as_unicode_text=True),
    )

    assert "$e_x+ f^y+ g_{prob}$" in rendered


def test_middle_to_markdown_drops_braces_for_single_char_groups() -> None:
    middle = Middle(
        pdf_info=[
            PageInfo(
                page_idx=0,
                page_size=(100, 100),
                para_blocks=[
                    Block(
                        type="text",
                        lines=[
                            Line(
                                spans=[
                                    Span(
                                        type="inline_equation",
                                        content=r"{P} + q_{R} + s^{T} + u_{prob}",
                                    )
                                ]
                            )
                        ],
                    )
                ],
            )
        ]
    )

    rendered = middle_to_markdown(
        middle,
        options=RenderOptions(render_latex_as_unicode_text=True),
    )

    assert "$P+ q_R+ s^T+ u_{prob}$" in rendered


def test_middle_to_markdown_collapses_spaces_inside_numeric_runs() -> None:
    middle = Middle(
        pdf_info=[
            PageInfo(
                page_idx=0,
                page_size=(100, 100),
                para_blocks=[
                    Block(
                        type="text",
                        lines=[
                            Line(
                                spans=[
                                    Span(
                                        type="inline_equation",
                                        content=r"(0 . 6 2 0 . 1 1)",
                                    )
                                ]
                            )
                        ],
                    )
                ],
            )
        ]
    )

    rendered = middle_to_markdown(middle)

    assert "$(0.620.11)$" in rendered
