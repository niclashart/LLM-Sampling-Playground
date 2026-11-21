"""Core generation logic for the LLM Sampling Playground.

The Streamlit app and the CLI are thin skins over this module. Everything
here is UI-free so it can be tested without running Streamlit.
"""

from dataclasses import dataclass
from functools import cache
from io import StringIO

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HISTORY_FIELDS = [
    "timestamp",
    "model",
    "prompt",
    "output",
    "temperature",
    "top_k",
    "top_p",
    "do_sample",
    "prompt_tokens",
    "generated_tokens",
]


def history_to_csv(history: list[dict]) -> str:
    """Render generation-history rows (see app.py) as a CSV string."""
    import csv

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=HISTORY_FIELDS)
    writer.writeheader()
    writer.writerows(history)
    return buffer.getvalue()


@dataclass(frozen=True)
class SamplingParams:
    """A single bundle of every sampling knob the UI can set."""

    do_sample: bool = True
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    max_new_tokens: int = 100
    seed: int | None = None


@dataclass(frozen=True)
class TokenUsage:
    """Token consumption of a single generation call."""

    prompt_tokens: int
    generated_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.generated_tokens


@dataclass(frozen=True)
class GenerationResult:
    """The generated continuation plus per-token probabilities for plotting."""

    text: str
    tokens: list[str]
    probabilities: list[float]
    usage: TokenUsage


@cache
def load_tokenizer(model_name: str):
    """Load just a tokenizer (no weights), cached — cheap enough to fan out
    across many models for the tokenizer-comparison chart."""
    return AutoTokenizer.from_pretrained(model_name)


@cache
def load_model(model_name: str):
    """Load a model + tokenizer pair, cached so each model loads once."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer


def build_generation_kwargs(params: SamplingParams) -> dict:
    """Map UI parameters to transformers generate() kwargs.

    In greedy mode (do_sample=False) the sampling parameters are dropped
    entirely, since transformers would warn that they have no effect.
    """
    kwargs = {
        "do_sample": params.do_sample,
        "max_new_tokens": params.max_new_tokens,
        "repetition_penalty": params.repetition_penalty,
        "num_return_sequences": 1,
    }
    if params.do_sample:
        kwargs.update(
            temperature=params.temperature,
            top_k=params.top_k,
            top_p=params.top_p,
        )
    return kwargs


def extract_token_probabilities(
    scores: tuple[torch.Tensor, ...],
    token_ids: torch.Tensor,
    vocab: dict[str, int],
) -> tuple[list[str], list[float]]:
    """Turn per-step generation scores into (token strings, probabilities).

    `scores` is one logits tensor per generated position, `token_ids` the
    ids the model actually picked (batch of 1). This is what powers the
    "how confident was the model at each token" chart.
    """
    id_to_token = {i: t for t, i in vocab.items()}
    tokens = [id_to_token[i] for i in token_ids[0].tolist()]
    probabilities = [
        torch.softmax(step_logits[0], dim=-1)[picked].item()
        for step_logits, picked in zip(scores, token_ids[0])
    ]
    return tokens, probabilities


def count_usage(tokenizer, prompt: str, new_token_ids: torch.Tensor) -> TokenUsage:
    """Count prompt vs. generated tokens for a single generation call."""
    prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=True))
    return TokenUsage(
        prompt_tokens=prompt_tokens,
        generated_tokens=new_token_ids.numel(),
    )


def compare_token_counts(prompt: str, tokenizers: dict) -> list[dict]:
    """Encode the same prompt with every tokenizer to compare token efficiency.

    Returns one row per tokenizer: {model, prompt_tokens}.
    """
    return [
        {"model": name, "prompt_tokens": len(tok.encode(prompt, add_special_tokens=True))}
        for name, tok in tokenizers.items()
    ]


def avg_tokens_by_model(history: list[dict]) -> dict[str, dict]:
    """Aggregate generation history into mean token usage per model."""
    sums: dict[str, dict] = {}
    for row in history:
        entry = sums.setdefault(
            row["model"],
            {"prompt_tokens": 0, "generated_tokens": 0, "runs": 0},
        )
        entry["prompt_tokens"] += row["prompt_tokens"]
        entry["generated_tokens"] += row["generated_tokens"]
        entry["runs"] += 1
    return {
        model: {
            "prompt_tokens": entry["prompt_tokens"] / entry["runs"],
            "generated_tokens": entry["generated_tokens"] / entry["runs"],
            "runs": entry["runs"],
        }
        for model, entry in sums.items()
    }


def comparison_summary_row(
    model: str, result: GenerationResult, elapsed_s: float, max_new_tokens: int
) -> dict:
    """One table row for an active multi-model comparison run.

    `hit_max_tokens` tells you whether the generated-token count is a real
    model stopping point (EOS) or just the limit you set — only uncapped
    runs make token-count comparisons meaningful.
    """
    return {
        "model": model,
        "output": result.text,
        "prompt_tokens": result.usage.prompt_tokens,
        "generated_tokens": result.usage.generated_tokens,
        "total_tokens": result.usage.total_tokens,
        "elapsed_s": round(elapsed_s, 2),
        "tokens_per_s": (
            round(result.usage.generated_tokens / elapsed_s, 1) if elapsed_s > 0 else 0.0
        ),
        "hit_max_tokens": result.usage.generated_tokens >= max_new_tokens,
    }


def generate(prompt: str, params: SamplingParams, model_name: str) -> GenerationResult:
    """Generate a continuation and return text + token-level probabilities."""
    model, tokenizer = load_model(model_name)

    if params.seed is not None:
        torch.manual_seed(params.seed)

    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(
        **inputs,
        **build_generation_kwargs(params),
        return_dict_in_generate=True,
        output_scores=True,
    )

    new_token_ids = output.sequences[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()

    tokens, probabilities = extract_token_probabilities(
        output.scores, new_token_ids.unsqueeze(0), tokenizer.get_vocab()
    )
    return GenerationResult(
        text=text,
        tokens=tokens,
        probabilities=probabilities,
        usage=count_usage(tokenizer, prompt, new_token_ids),
    )
