import pytest
import torch

from generator import (
    GenerationResult,
    SamplingParams,
    TokenUsage,
    avg_tokens_by_model,
    build_generation_kwargs,
    compare_token_counts,
    comparison_summary_row,
    count_usage,
    extract_token_probabilities,
    history_to_csv,
)


class TestBuildGenerationKwargs:
    def test_sampling_mode_includes_sampling_parameters(self):
        params = SamplingParams(
            do_sample=True,
            temperature=1.3,
            top_k=40,
            top_p=0.7,
            repetition_penalty=1.2,
            max_new_tokens=80,
        )
        kwargs = build_generation_kwargs(params)
        assert kwargs == {
            "do_sample": True,
            "temperature": 1.3,
            "top_k": 40,
            "top_p": 0.7,
            "repetition_penalty": 1.2,
            "max_new_tokens": 80,
            "num_return_sequences": 1,
        }

    def test_greedy_mode_ignores_sampling_parameters(self):
        params = SamplingParams(
            do_sample=False,
            temperature=1.3,
            top_k=40,
            top_p=0.7,
            repetition_penalty=1.2,
            max_new_tokens=80,
        )
        kwargs = build_generation_kwargs(params)
        assert kwargs == {
            "do_sample": False,
            "repetition_penalty": 1.2,
            "max_new_tokens": 80,
            "num_return_sequences": 1,
        }


class TestExtractTokenProbabilities:
    def test_returns_decoded_tokens_and_probabilities(self):
        vocab = {"Hello": 0, "world": 1, "!": 2}
        scores = torch.log(torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3]]))
        scores = (scores[0].unsqueeze(0), scores[1].unsqueeze(0))
        token_ids = torch.tensor([[0, 1]])

        tokens, probabilities = extract_token_probabilities(
            scores, token_ids, vocab
        )

        assert tokens == ["Hello", "world"]
        assert probabilities == pytest.approx([0.7, 0.6])


class StubTokenizer:
    """Char-level stand-in: one token per character."""

    def encode(self, text, add_special_tokens=True):
        tokens = list(text)
        if add_special_tokens:
            return [0, *tokens]
        return tokens


class TestCountUsage:
    def test_counts_prompt_and_generated_tokens(self):
        usage = count_usage(
            StubTokenizer(), prompt="abc", new_token_ids=torch.tensor([[0, 1]])
        )
        assert usage == TokenUsage(prompt_tokens=4, generated_tokens=2)

    def test_adds_special_tokens_to_prompt_count(self):
        usage = count_usage(
            StubTokenizer(), prompt="ab", new_token_ids=torch.tensor([[0]])
        )
        assert usage == TokenUsage(prompt_tokens=3, generated_tokens=1)


class TestHistoryToCsv:
    def test_renders_history_rows_as_csv(self):
        history = [
            {
                "timestamp": "2026-09-04 12:00:00",
                "model": "gpt2",
                "prompt": 'Say "hi"',
                "output": "hi there",
                "temperature": 1.0,
                "top_k": 50,
                "top_p": 0.9,
                "do_sample": True,
                "prompt_tokens": 3,
                "generated_tokens": 2,
            }
        ]
        csv_text = history_to_csv(history)
        lines = csv_text.strip().splitlines()
        assert lines[0] == (
            "timestamp,model,prompt,output,temperature,top_k,top_p,"
            "do_sample,prompt_tokens,generated_tokens"
        )
        assert lines[1] == (
            '2026-09-04 12:00:00,gpt2,"Say ""hi""",hi there,1.0,50,0.9,'
            "True,3,2"
        )

    def test_empty_history_is_header_only(self):
        assert history_to_csv([]).strip() == (
            "timestamp,model,prompt,output,temperature,top_k,top_p,"
            "do_sample,prompt_tokens,generated_tokens"
        )


class TestCompareTokenCounts:
    def test_encodes_same_prompt_with_every_tokenizer(self):
        class PlainTokenizer(StubTokenizer):
            def encode(self, text, add_special_tokens=True):
                return list(text)

        tokenizers = {
            "special-token-model": StubTokenizer(),
            "plain-char-model": PlainTokenizer(),
        }

        rows = compare_token_counts("abc", tokenizers)
        assert rows == [
            {"model": "special-token-model", "prompt_tokens": 4},
            {"model": "plain-char-model", "prompt_tokens": 3},
        ]


class TestAvgTokensByModel:
    def test_averages_usage_per_model(self):
        history = [
            {"model": "gpt2", "prompt_tokens": 4, "generated_tokens": 10},
            {"model": "gpt2", "prompt_tokens": 4, "generated_tokens": 20},
            {"model": "lfm", "prompt_tokens": 6, "generated_tokens": 30},
        ]
        assert avg_tokens_by_model(history) == {
            "gpt2": {"prompt_tokens": 4.0, "generated_tokens": 15.0, "runs": 2},
            "lfm": {"prompt_tokens": 6.0, "generated_tokens": 30.0, "runs": 1},
        }

    def test_empty_history_is_empty(self):
        assert avg_tokens_by_model([]) == {}


class TestComparisonSummaryRow:
    def _result(self) -> GenerationResult:
        return GenerationResult(
            text="hello",
            tokens=["hello"],
            probabilities=[0.9],
            usage=TokenUsage(prompt_tokens=5, generated_tokens=1),
        )

    def test_summarizes_one_comparison_run(self):
        row = comparison_summary_row("gpt2", self._result(), elapsed_s=2.0, max_new_tokens=60)
        assert row == {
            "model": "gpt2",
            "output": "hello",
            "prompt_tokens": 5,
            "generated_tokens": 1,
            "total_tokens": 6,
            "elapsed_s": 2.0,
            "tokens_per_s": 0.5,
            "hit_max_tokens": False,
        }

    def test_flags_runs_that_reached_the_token_limit(self):
        result = GenerationResult(
            text="a" * 60,
            tokens=["a"] * 60,
            probabilities=[0.9] * 60,
            usage=TokenUsage(prompt_tokens=5, generated_tokens=60),
        )
        row = comparison_summary_row("gpt2", result, elapsed_s=2.0, max_new_tokens=60)
        assert row["hit_max_tokens"] is True
