from __future__ import annotations

from importlib import import_module

from ocrstruct.middle_to_html import default_html_header_path, markdown_to_html


def test_markdown_to_html_returns_none_without_pandoc(monkeypatch) -> None:
    middle_to_html_module = import_module("ocrstruct.middle_to_html")
    monkeypatch.setattr(middle_to_html_module.shutil, "which", lambda _: None)

    assert markdown_to_html("# hello") is None


def test_default_html_header_uses_wrapped_fixed_width_tables() -> None:
    header_path = default_html_header_path()

    assert header_path is not None
    header = header_path.read_text(encoding="utf-8")
    assert "table-layout: fixed;" in header
    assert "overflow-wrap: anywhere;" in header
    assert "overflow-x: auto;" not in header
