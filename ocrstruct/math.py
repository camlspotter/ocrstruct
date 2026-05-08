from __future__ import annotations

import re


_LATEX_TO_UNICODE = {
    r"\alpha": "α",
    r"\beta": "β",
    r"\gamma": "γ",
    r"\delta": "δ",
    r"\epsilon": "ε",
    r"\varepsilon": "ε",
    r"\zeta": "ζ",
    r"\eta": "η",
    r"\theta": "θ",
    r"\vartheta": "ϑ",
    r"\iota": "ι",
    r"\kappa": "κ",
    r"\lambda": "λ",
    r"\mu": "μ",
    r"\nu": "ν",
    r"\xi": "ξ",
    r"\pi": "π",
    r"\varpi": "ϖ",
    r"\rho": "ρ",
    r"\varrho": "ϱ",
    r"\sigma": "σ",
    r"\varsigma": "ς",
    r"\tau": "τ",
    r"\upsilon": "υ",
    r"\phi": "φ",
    r"\varphi": "ϕ",
    r"\chi": "χ",
    r"\psi": "ψ",
    r"\omega": "ω",
    r"\Gamma": "Γ",
    r"\Delta": "Δ",
    r"\Theta": "Θ",
    r"\Lambda": "Λ",
    r"\Xi": "Ξ",
    r"\Pi": "Π",
    r"\Sigma": "Σ",
    r"\Phi": "Φ",
    r"\Psi": "Ψ",
    r"\Omega": "Ω",
    r"\times": "×",
    r"\cdot": "·",
    r"\pm": "±",
    r"\mp": "∓",
    r"\neq": "≠",
    r"\le": "≤",
    r"\leq": "≤",
    r"\ge": "≥",
    r"\geq": "≥",
    r"\ll": "≪",
    r"\gg": "≫",
    r"\approx": "≈",
    r"\equiv": "≡",
    r"\sim": "∼",
    r"\to": "→",
    r"\rightarrow": "→",
    r"\leftarrow": "←",
    r"\leftrightarrow": "↔",
    r"\Rightarrow": "⇒",
    r"\Leftarrow": "⇐",
    r"\Leftrightarrow": "⇔",
    r"\mapsto": "↦",
    r"\infty": "∞",
    r"\partial": "∂",
    r"\nabla": "∇",
    r"\sum": "∑",
    r"\prod": "∏",
    r"\int": "∫",
    r"\oint": "∮",
    r"\in": "∈",
    r"\notin": "∉",
    r"\ni": "∋",
    r"\subset": "⊂",
    r"\subseteq": "⊆",
    r"\supset": "⊃",
    r"\supseteq": "⊇",
    r"\cup": "∪",
    r"\cap": "∩",
    r"\forall": "∀",
    r"\exists": "∃",
    r"\neg": "¬",
    r"\land": "∧",
    r"\lor": "∨",
}

def render_math_text(
    text: str,
    *,
    render_latex_as_unicode_text: bool,
    display: bool,
) -> str:
    normalized = text.strip()
    if normalized.startswith("$$") and normalized.endswith("$$"):
        normalized = normalized.removeprefix("$$").removesuffix("$$").strip()
    if normalized.startswith("$") and normalized.endswith("$"):
        normalized = normalized.removeprefix("$").removesuffix("$").strip()
    normalized = normalize_latex_math_spacing(normalized)
    if render_latex_as_unicode_text:
        normalized = latex_to_unicode_text(normalized)
    if display:
        return f"$$\n{normalized}\n$$"
    return f"${normalized}$"


def normalize_latex_math_spacing(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    normalized = re.sub(r"([_^])\s+", r"\1", normalized)
    normalized = re.sub(r"\s+([_^])", r"\1", normalized)
    normalized = re.sub(r"([{}()])\s+", r"\1", normalized)
    normalized = re.sub(r"\s+([{}()])", r"\1", normalized)
    normalized = re.sub(r"(?<=[0-9.,()])\s+(?=[0-9.,()])", "", normalized)
    normalized = re.sub(r"\\([A-Za-z]+)\s+(?=[^A-Za-z\s])", r"\\\1", normalized)
    normalized = re.sub(r"\\([A-Za-z]+)\s+(?=\\[A-Za-z]+)", r"\\\1 ", normalized)
    normalized = re.sub(
        r"\{((?:[A-Za-z0-9]\s+){1,}[A-Za-z0-9])\}",
        lambda m: f"{{{re.sub(r'\s+', '', m.group(1))}}}",
        normalized,
    )
    return normalized


def latex_to_unicode_text(text: str) -> str:
    rendered = text
    for macro in ("mathrm", "mathit", "mathbb"):
        rendered = re.sub(rf"\\{macro}\{{([^{{}}]+)\}}", r"{\1}", rendered)
    for latex, unicode_text in _LATEX_TO_UNICODE.items():
        rendered = rendered.replace(latex, unicode_text)
    rendered = re.sub(r"\{\{([^{}]+)\}\}", r"{\1}", rendered)
    rendered = re.sub(r"\{([A-Za-z0-9])\}", r"\1", rendered)
    return rendered
