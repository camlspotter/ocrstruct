from __future__ import annotations

from pathlib import Path
import subprocess

from ocrstruct.html import elements_to_html, markdown_to_html
from ocrstruct.table import encode_html_table_eq_tokens
from ocrstruct.types import Element


def test_markdown_to_html_returns_none_without_pandoc(monkeypatch) -> None:
    monkeypatch.setattr("ocrstruct.html.shutil.which", lambda _: None)

    assert markdown_to_html("# hello") is None


def test_elements_to_html_postprocesses_math(monkeypatch, tmp_path: Path) -> None:
    def fake_run(command: list[str], check: bool) -> subprocess.CompletedProcess[str]:
        assert check is True
        output_path = Path(command[command.index("-o") + 1])
        encoded = encode_html_table_eq_tokens("<eq>x + y</eq>")
        output_path.write_text(
            f"<html><body><eq>z^2</eq>{encoded}</body></html>",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("ocrstruct.html.shutil.which", lambda _: "/usr/bin/pandoc")
    monkeypatch.setattr("ocrstruct.html.subprocess.run", fake_run)

    elements = [
        Element(
            kind="text",
            text="hello",
            subkind=None,
            image_path=None,
            alt=None,
            description=None,
            level=None,
            loc=None,
        )
    ]

    html = elements_to_html(elements, header_path=tmp_path / "style.html")

    assert html is not None
    assert html.count('<span class="math inline">') == 2
    assert "\\(z^2\\)" in html
    assert "\\(x + y\\)" in html
