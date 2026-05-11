from __future__ import annotations

from ocrstruct.html import markdown_to_html


def test_markdown_to_html_returns_none_without_pandoc(monkeypatch) -> None:
    monkeypatch.setattr("ocrstruct.html.shutil.which", lambda _: None)

    assert markdown_to_html("# hello") is None
