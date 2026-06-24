from __future__ import annotations

from pathlib import Path

from ocrstruct.image_understanding import (
    build_images_file,
    compute_middle_json_sha256,
    ImageRef,
    ImageUnderstanding,
    ImageUnderstandingRunResult,
    load_images_file_json,
    ModelPricing,
    RunStatus,
    ScreeningRecord,
    ScreeningResult,
    ScreeningRunView,
    ScreeningSource,
    UnderstandingRecord,
    UnderstandingRunView,
    iter_understanding_records_from_screening,
    load_completed_understanding_keys,
    load_pricing_overrides,
    load_screening_records_jsonl,
    load_understanding_records_jsonl,
    merge_images_into_middle,
    merge_understanding_into_middle,
    image_refs_from_middle,
)
from ocrstruct.middle import Block, Line, Middle, PageInfo, Span
from ocrstruct.result import Result


def _screening_record() -> ScreeningRecord:
    ref = ImageRef(
        pdf_path="/tmp/sample.pdf",
        middle_json_path="/tmp/middle.json",
        page_idx=0,
        block_index=1,
        block_type="image",
        image_path="sample.png",
    )
    return ScreeningRecord(
        ref=ref,
        model="screening-model",
        thinking=False,
        resolved_thinking=False,
        base_url="http://localhost:18000/v1",
        started_at="2026-05-01T00:00:00+00:00",
        latency_sec=1.0,
        status=RunStatus(ok=True),
        run=ScreeningRunView(
            kind="diagram",
            rag_value="high",
            detail_level="short",
            notes="sample",
            raw_text='{"kind":"diagram"}',
        ),
    )


def test_load_screening_records_jsonl_filters_failed_and_thinking(tmp_path: Path) -> None:
    ok_record = _screening_record()
    failed_record = ok_record.model_copy(update={"status": RunStatus(ok=False, error="boom")})
    thinking_record = ok_record.model_copy(update={"thinking": True})
    path = tmp_path / "screening.jsonl"
    path.write_text(
        "\n".join(
            [
                ok_record.model_dump_json(),
                failed_record.model_dump_json(),
                thinking_record.model_dump_json(),
            ]
        ),
        encoding="utf-8",
    )

    records = load_screening_records_jsonl(path, screening_thinking=False)

    assert len(records) == 1
    assert records[0].thinking is False


def test_load_completed_understanding_keys_keeps_success_rows_only(tmp_path: Path) -> None:
    screening_record = _screening_record()
    assert screening_record.run is not None
    success = UnderstandingRecord(
        ref=screening_record.ref,
        screening=ScreeningSource(
            model=screening_record.model,
            thinking=screening_record.thinking,
            resolved_thinking=screening_record.resolved_thinking,
            base_url=screening_record.base_url,
            started_at=screening_record.started_at,
            latency_sec=screening_record.latency_sec,
            run=screening_record.run,
        ),
        model="understanding-model",
        thinking=False,
        resolved_thinking=False,
        base_url=None,
        started_at="2026-05-01T00:00:01+00:00",
        latency_sec=2.0,
        status=RunStatus(ok=True),
        run=None,
    )
    failed = success.model_copy(update={"status": RunStatus(ok=False, error="boom")})
    path = tmp_path / "understanding.jsonl"
    path.write_text(
        "\n".join([success.model_dump_json(), failed.model_dump_json()]),
        encoding="utf-8",
    )

    keys = load_completed_understanding_keys(path)

    assert len(keys) == 1


def test_load_understanding_records_jsonl_filters_failed_rows(tmp_path: Path) -> None:
    screening_record = _screening_record()
    assert screening_record.run is not None
    success = UnderstandingRecord(
        ref=screening_record.ref,
        screening=ScreeningSource(
            model=screening_record.model,
            thinking=screening_record.thinking,
            resolved_thinking=screening_record.resolved_thinking,
            base_url=screening_record.base_url,
            started_at=screening_record.started_at,
            latency_sec=screening_record.latency_sec,
            run=screening_record.run,
        ),
        model="understanding-model",
        thinking=False,
        resolved_thinking=False,
        base_url=None,
        started_at="2026-05-01T00:00:01+00:00",
        latency_sec=2.0,
        status=RunStatus(ok=True),
        run=UnderstandingRunView(
            kind="diagram",
            rag_value="high",
            detail_level="short",
            keywords=["k"],
            notes=None,
            short_description="short",
            long_description=None,
            raw_text='{"keywords":["k"]}',
        ),
    )
    failed = success.model_copy(update={"status": RunStatus(ok=False, error="boom"), "run": None})
    path = tmp_path / "understanding.jsonl"
    path.write_text(
        "\n".join([success.model_dump_json(), failed.model_dump_json()]),
        encoding="utf-8",
    )

    records = load_understanding_records_jsonl(path)

    assert len(records) == 1
    assert records[0].model == "understanding-model"


def test_iter_understanding_records_from_screening_skips_existing_and_yields_success(
    monkeypatch,
) -> None:
    screening_record = _screening_record()

    def fake_run(
        ref: ImageRef,
        screening: ScreeningResult,
        *,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        pricing: ModelPricing,
        thinking: bool = False,
    ) -> ImageUnderstandingRunResult:
        assert ref == screening_record.ref
        assert screening.kind == "diagram"
        assert model == "understanding-model"
        assert pricing.input_per_million_usd == 1.0
        assert thinking is False
        return ImageUnderstandingRunResult(
            model=model,
            base_url=base_url,
            started_at="2026-05-01T00:00:02+00:00",
            raw_text='{"keywords":["k"]}',
            result=ImageUnderstanding(
                ref=ref,
                kind=screening.kind,
                rag_value=screening.rag_value,
                detail_level=screening.detail_level,
                keywords=["k"],
                short_description="short",
            ),
        )

    monkeypatch.setattr(
        "ocrstruct.image_understanding.image_understanding_run_from_screening",
        fake_run,
    )

    records = list(
        iter_understanding_records_from_screening(
            [screening_record],
            model="understanding-model",
            pricing=ModelPricing(input_per_million_usd=1.0, output_per_million_usd=2.0),
        )
    )

    assert len(records) == 1
    assert records[0].status.ok is True
    assert records[0].run is not None
    assert records[0].run.short_description == "short"

    skipped = list(
        iter_understanding_records_from_screening(
            [screening_record],
            model="understanding-model",
            pricing=ModelPricing(input_per_million_usd=1.0, output_per_million_usd=2.0),
            existing_keys={
                (
                    "understanding-model",
                    False,
                    "screening-model",
                    False,
                    ("/tmp/middle.json", 0, 1, "sample.png"),
                )
            },
        )
    )

    assert skipped == []


def test_load_pricing_overrides_reads_model_map(tmp_path: Path) -> None:
    path = tmp_path / "pricing.json"
    path.write_text(
        '{"demo-model":{"input_per_million_usd":1.5,"output_per_million_usd":2.5}}',
        encoding="utf-8",
    )

    pricing = load_pricing_overrides(path)

    assert pricing["demo-model"].input_per_million_usd == 1.5


def test_load_image_refs_from_middle_json_reads_result_wrapper(tmp_path: Path) -> None:
    middle_path = tmp_path / "middle.json"
    result = Result(
        source_path="dummy",
        middle=Middle(
            pdf_info=[
                PageInfo(
                    page_idx=0,
                    page_size=(100, 100),
                    para_blocks=[
                        Block(
                            type="image",
                            index=7,
                            blocks=[
                                Block(
                                    type="image_body",
                                    lines=[Line(spans=[Span(type="image", image_path="sample.png")])],
                                )
                            ],
                        )
                    ],
                )
            ]
        ),
        extracted_by="mineru/pipeline",
    )
    result.save_json(middle_path)
    refs = image_refs_from_middle(result.middle, pdf_path= 'dummy.pdf', middle_json_path= str(middle_path))

    assert len(refs) == 1
    assert refs[0].middle_json_path == str(middle_path)
    assert refs[0].image_path == "sample.png"
    assert refs[0].block_index == 7


def test_merge_understanding_into_middle_attaches_summary_to_image_span() -> None:
    screening_record = _screening_record()
    assert screening_record.run is not None
    understanding_record = UnderstandingRecord(
        ref=screening_record.ref,
        screening=ScreeningSource(
            model=screening_record.model,
            thinking=screening_record.thinking,
            resolved_thinking=screening_record.resolved_thinking,
            base_url=screening_record.base_url,
            started_at=screening_record.started_at,
            latency_sec=screening_record.latency_sec,
            run=screening_record.run,
        ),
        model="understanding-model",
        thinking=False,
        resolved_thinking=False,
        base_url=None,
        started_at="2026-05-01T00:00:02+00:00",
        latency_sec=1.5,
        status=RunStatus(ok=True),
        run=UnderstandingRunView(
            kind="diagram",
            rag_value="high",
            detail_level="short",
            keywords=["workflow"],
            notes="readable",
            short_description="図の説明",
            long_description=None,
            raw_text='{"keywords":["workflow"]}',
        ),
    )
    middle = Middle(
        pdf_info=[
            PageInfo(
                page_idx=0,
                page_size=(100, 100),
                para_blocks=[
                    Block(
                        type="image",
                        index=1,
                        blocks=[
                            Block(
                                type="image_body",
                                lines=[Line(spans=[Span(type="image", image_path="sample.png")])],
                            )
                        ],
                    )
                ],
            )
        ]
    )

    merged = merge_understanding_into_middle(middle, [understanding_record])

    target_span = merged.pdf_info[0].para_blocks[0].blocks[0].lines[0].spans[0]
    assert target_span.image_understanding is not None
    assert target_span.image_understanding.short_description == "図の説明"
    assert target_span.image_understanding.model == "understanding-model"
    assert target_span.image_understanding.screening_model == "screening-model"
    assert middle.pdf_info[0].para_blocks[0].blocks[0].lines[0].spans[0].image_understanding is None


def _understanding_record() -> UnderstandingRecord:
    screening_record = _screening_record()
    assert screening_record.run is not None
    return UnderstandingRecord(
        ref=screening_record.ref.model_copy(
            update={
                "caption": "Figure 1",
                "nearby_text_before": "before",
                "nearby_text_after": "after",
                "section_title": "Section A",
            }
        ),
        screening=ScreeningSource(
            model=screening_record.model,
            thinking=screening_record.thinking,
            resolved_thinking=screening_record.resolved_thinking,
            base_url=screening_record.base_url,
            started_at=screening_record.started_at,
            latency_sec=screening_record.latency_sec,
            run=screening_record.run,
        ),
        model="understanding-model",
        thinking=False,
        resolved_thinking=False,
        base_url=None,
        started_at="2026-05-01T00:00:02+00:00",
        latency_sec=1.5,
        status=RunStatus(ok=True),
        run=UnderstandingRunView(
            kind="diagram",
            rag_value="high",
            detail_level="short",
            keywords=["workflow"],
            notes="readable",
            short_description="図の説明",
            long_description="長い説明",
            raw_text='{"keywords":["workflow"]}',
        ),
    )


def test_build_images_file_embeds_middle_hash_and_context(tmp_path: Path) -> None:
    middle_path = tmp_path / "middle.json"
    middle_path.write_text('{"hello":"world"}', encoding="utf-8")

    images_file = build_images_file(
        [_understanding_record()],
        middle_json_path=middle_path,
    )

    assert images_file.middle_json_sha256 == compute_middle_json_sha256(middle_path)
    assert len(images_file.items) == 1
    assert images_file.items[0].ref.caption == "Figure 1"
    assert images_file.items[0].ref.section_title == "Section A"
    assert images_file.items[0].understanding.short_description == "図の説明"
    assert images_file.items[0].screening.model == "screening-model"


def test_load_images_file_json_verifies_middle_hash(tmp_path: Path) -> None:
    middle_path = tmp_path / "middle.json"
    middle_path.write_text('{"hello":"world"}', encoding="utf-8")
    images_file = build_images_file(
        [_understanding_record()],
        middle_json_path=middle_path,
    )
    images_path = tmp_path / "images.json"
    images_file.save_json(images_path)

    loaded = load_images_file_json(images_path, middle_json_path=middle_path)
    assert loaded.middle_json_sha256 == images_file.middle_json_sha256

    other_middle_path = tmp_path / "other-middle.json"
    other_middle_path.write_text('{"hello":"other"}', encoding="utf-8")
    try:
        load_images_file_json(images_path, middle_json_path=other_middle_path)
    except ValueError as error:
        assert "does not match" in str(error)
    else:
        raise AssertionError("expected hash mismatch to raise ValueError")


def test_merge_images_into_middle_attaches_summary_to_image_span(tmp_path: Path) -> None:
    understanding_record = _understanding_record()
    middle_path = tmp_path / "middle.json"
    middle_path.write_text('{"hello":"world"}', encoding="utf-8")
    understanding_record = understanding_record.model_copy(
        update={
            "ref": understanding_record.ref.model_copy(
                update={"middle_json_path": str(middle_path)}
            )
        }
    )
    images_file = build_images_file(
        [understanding_record],
        middle_json_path=middle_path,
    )
    middle = Middle(
        pdf_info=[
            PageInfo(
                page_idx=0,
                page_size=(100, 100),
                para_blocks=[
                    Block(
                        type="image",
                        index=1,
                        blocks=[
                            Block(
                                type="image_body",
                                lines=[Line(spans=[Span(type="image", image_path="sample.png")])],
                            )
                        ],
                    )
                ],
            )
        ]
    )

    merged = merge_images_into_middle(middle, images_file)

    target_span = merged.pdf_info[0].para_blocks[0].blocks[0].lines[0].spans[0]
    assert target_span.image_understanding is not None
    assert target_span.image_understanding.short_description == "図の説明"
    assert target_span.image_understanding.model == "understanding-model"
    assert target_span.image_understanding.screening_model == "screening-model"
