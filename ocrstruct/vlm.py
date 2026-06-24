from __future__ import annotations

import mimetypes
import base64
from typing import Any, Iterable, Literal, TypeAlias, cast
from pathlib import Path
import json
from pydantic import BaseModel, ConfigDict


class TokenUsage(BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


class PriceEstimate(BaseModel):
    input_cost_usd: float | None = None
    output_cost_usd: float | None = None
    total_cost_usd: float | None = None


class ModelPricing(BaseModel, frozen= True):
    input_per_million_usd: float
    output_per_million_usd: float


class VLM(BaseModel):
    model: str
    thinking: bool


class VLMConfig(BaseModel, frozen= True):
    model: str
    thinking: bool
    base_url: str | None = None
    api_key: str | None = None
    pricing: ModelPricing

    @property
    def vlm(self) -> VLM:
        return VLM(model= self.model, thinking= self.thinking)


class StructuredOutputModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


DEFAULT_MODEL_PRICING: dict[str, ModelPricing] = {
    "Qwen/Qwen3.6-35B-A3B-FP8": ModelPricing(
        input_per_million_usd=0.0,
        output_per_million_usd=0.0,
    ),
    "Qwen/Qwen3.6-27B-FP8": ModelPricing(
        input_per_million_usd=0.0,
        output_per_million_usd=0.0,
    ),
    "google/gemma-4-26B-A4B-it": ModelPricing(
        input_per_million_usd=0.0,
        output_per_million_usd=0.0,
    ),
    "gpt-5.4": ModelPricing(input_per_million_usd=2.5, output_per_million_usd=15.0),
    "gpt-5.4-mini": ModelPricing(input_per_million_usd=0.75, output_per_million_usd=4.5),
    "gpt-5.4-nano": ModelPricing(input_per_million_usd=0.2, output_per_million_usd=1.25),
    "gpt-5.2": ModelPricing(input_per_million_usd=1.75, output_per_million_usd=14.0),
    "gpt-5": ModelPricing(input_per_million_usd=1.25, output_per_million_usd=10.0),
    "gpt-5-mini": ModelPricing(input_per_million_usd=0.25, output_per_million_usd=2.0),
    "gpt-5-nano": ModelPricing(input_per_million_usd=0.05, output_per_million_usd=0.4),
    "gpt-4.1": ModelPricing(input_per_million_usd=2.0, output_per_million_usd=8.0),
    "gpt-4.1-mini": ModelPricing(input_per_million_usd=0.4, output_per_million_usd=1.6),
    "gpt-4o": ModelPricing(input_per_million_usd=2.5, output_per_million_usd=10.0),
    "gpt-4o-mini": ModelPricing(input_per_million_usd=0.15, output_per_million_usd=0.6),
}


def _apply_thinking_option(
    request: dict[str, object],
    *,
    model: str,
    thinking: bool
) -> None:
    '''Ugly function to support various models with different thinking options'''
    supports_chat_template_thinking = model.startswith(("Qwen/", "google/gemma-4"))
    if model.startswith(("gpt-5.4", "gpt-5.5", "gpt-5.1", "gpt-5")):
        if thinking:
            request["reasoning_effort"] = 'low'
        else:
            if model.startswith(("gpt-5.4", "gpt-5.5", "gpt-5.1")):
                request["reasoning_effort"] = "none"
            elif model.startswith("gpt-5"):
                request["reasoning_effort"] = "minimal"
    elif supports_chat_template_thinking:
        if thinking:
            request["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": True},
            }
        else:
            request["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": False},
            }


def _openai_strict_json_schema(schema_model: type[BaseModel]) -> dict[str, object]:
    schema = cast(dict[str, object], schema_model.model_json_schema())
    properties = cast(dict[str, object], schema.get("properties", {}))
    if properties:
        schema["required"] = list(properties.keys())
    return schema


def image_json_request(
    *,
    vlm: VLM,
    prompt: str,
    context_text: str,
    image_data_url: str,
    schema_name: str,
    schema_model: type[BaseModel],
) -> dict[str, object]:
    request: dict[str, object] = {
        "model": vlm.model,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": _openai_strict_json_schema(schema_model),
            },
        },
        "messages": [
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": context_text,
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_url},
                    },
                ],
            },
        ],
    }
    _apply_thinking_option(request, model=vlm.model, thinking=vlm.thinking)
    return request


def _usage_from_completion(completion: object) -> TokenUsage | None:
    usage = getattr(completion, "usage", None)
    if usage is None:
        return None
    input_tokens = getattr(usage, "prompt_tokens", None)
    output_tokens = getattr(usage, "completion_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )


def _estimate_price(
    usage: TokenUsage,
    pricing: ModelPricing | None,
) -> PriceEstimate | None:
    if pricing is None:
        return None
    input_cost_usd: float | None = None
    output_cost_usd: float | None = None
    if usage.input_tokens is not None:
        input_cost_usd = usage.input_tokens * pricing.input_per_million_usd / 1_000_000
    if usage.output_tokens is not None:
        output_cost_usd = usage.output_tokens * pricing.output_per_million_usd / 1_000_000
    if input_cost_usd is None and output_cost_usd is None:
        return None
    total_cost_usd = (input_cost_usd or 0.0) + (output_cost_usd or 0.0)
    return PriceEstimate(
        input_cost_usd=input_cost_usd,
        output_cost_usd=output_cost_usd,
        total_cost_usd=total_cost_usd,
    )

def estimate_price_from_completion(
    completion: object,
    pricing: ModelPricing | None,
) -> tuple[TokenUsage | None, PriceEstimate | None]:
    if usage := _usage_from_completion(completion):
        return usage, _estimate_price(usage, pricing)
    else:
        return None, None


def load_pricing_overrides(path: str | Path | None) -> dict[str, ModelPricing]:
    if path is None:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Pricing override must be a JSON object: {path}")
    out: dict[str, ModelPricing] = {}
    for model, value in data.items():
        out[model] = ModelPricing.model_validate(value)
    return out


def pricing_for_model(
    model: str,
    pricing_overrides: dict[str, ModelPricing] | None = None,
) -> ModelPricing:
    overrides = pricing_overrides or {}
    pricing = overrides.get(model, DEFAULT_MODEL_PRICING.get(model))
    if pricing is None:
        raise ValueError(
            "Unknown model pricing: "
            f"{model}. Add it to DEFAULT_MODEL_PRICING or pass --pricing-json."
        )
    return pricing


def image_data_url(image_file : Path) -> str:
    mime_type, _encoding = mimetypes.guess_type(image_file.name)
    if mime_type is None:
        mime_type = "application/octet-stream"
    raw = image_file.read_bytes()
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"
