from __future__ import annotations

from ocrstruct.middle import Block, Line, Middle, PageInfo, Result, Span


def test_middle_validation_replaces_broken_surrogates_in_content() -> None:
    middle = Middle.model_validate(
        {
            "pdf_info": [
                {
                    "page_idx": 0,
                    "page_size": (100, 100),
                    "para_blocks": [
                        {
                            "type": "text",
                            "lines": [
                                {
                                    "spans": [
                                        {
                                            "type": "text",
                                            "content": "\ud840abc",
                                        }
                                    ]
                                }
                            ],
                        }
                    ],
                }
            ]
        }
    )

    span = middle.pdf_info[0].para_blocks[0].lines[0].spans[0]
    assert span.content == "\ufffdabc"


def test_result_validation_replaces_broken_surrogates_in_metadata_keys() -> None:
    result = Result.model_validate(
        {
            "middle_json": {
                "pdf_info": [
                    {
                        "page_idx": 0,
                        "page_size": (100, 100),
                    }
                ],
                "_header_text_first_page": {
                    "\ud840broken": 0,
                },
            },
            "extracted_by": "mineru/pipeline",
        }
    )

    assert result.middle_json.header_text_first_page == {"\ufffdbroken": 0}


def test_result_save_json_handles_repaired_surrogates(tmp_path) -> None:
    result = Result(
        middle_json=Middle(
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
                                        Span(type="text", content="\ud840abc"),
                                    ]
                                )
                            ],
                        )
                    ],
                )
            ]
        ),
        extracted_by="mineru/pipeline",
    )

    out_path = tmp_path / "middle.json"
    result.save_json(out_path)

    saved = out_path.read_text(encoding="utf-8")
    assert "\ud840" not in saved
    assert "\ufffdabc" in saved
