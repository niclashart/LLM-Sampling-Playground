"""CLI demo: same generation logic as the Streamlit app, no UI required.

Usage: python cli.py --temperature 1.3 "The future of AI is"
"""

import argparse

from generator import SamplingParams, generate


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample text from GPT-2 models")
    parser.add_argument("prompt", nargs="?", default="The meaning of life is")
    parser.add_argument("--model", default="gpt2-medium")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--max-new-tokens", type=int, default=60)
    parser.add_argument("--greedy", action="store_true", help="Disable sampling")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    params = SamplingParams(
        do_sample=not args.greedy,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )
    result = generate(args.prompt, params, args.model)

    print(f"{args.prompt}{result.text}\n")
    mode = "greedy" if params.do_sample is False else "sampling"
    print(f"[mode={mode} temp={params.temperature} top_k={params.top_k} "
          f"top_p={params.top_p} seed={params.seed}]")


if __name__ == "__main__":
    main()
