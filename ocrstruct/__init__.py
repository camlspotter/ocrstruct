from ocrstruct.html import elements_to_html, markdown_to_html
from pathlib import Path

from ocrstruct.middle_to_elements import middle_to_elements
from ocrstruct.pdf import convert_pdf_to_middle, extract_pdf_link_regions, convert_pdf_to_elements
from ocrstruct.types import BBox, Element, LinkRegion, Location, elements_to_markdown


__all__ = [
    "BBox",
    "Element",
    "LinkRegion",
    "Location",
    "convert_pdf_to_middle",
    "convert_pdf_to_elements",
    "elements_to_html",
    "elements_to_markdown",
    "extract_pdf_link_regions",
    "markdown_to_html",
    "middle_to_elements",
]
