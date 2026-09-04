import datetime as dt
import time

import streamlit as st

from generator import (
    SamplingParams,
    avg_tokens_by_model,
    comparison_summary_row,
    compare_token_counts,
    generate,
    history_to_csv,
    load_model,
    load_tokenizer,
)

st.set_page_config(
    page_title="LLM Sampling Playground",
    page_icon="🎲",
    layout="wide",
)

MODELS = {
    "GPT-2 (124M, 2019)": "gpt2",
    "GPT-2 Medium (355M, 2019)": "gpt2-medium",
    "DistilGPT-2 (82M, 2019)": "distilgpt2",
    "SmolLM2 360M (2024)": "HuggingFaceTB/SmolLM2-360M",
    "Qwen2.5 0.5B (2024)": "Qwen/Qwen2.5-0.5B",
    "TinyLlama 1.1B (2024)": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    "LFM2.5 350M (2026)": "LiquidAI/LFM2.5-350M",
    "LFM2.5 1.2B Instruct (2026)": "LiquidAI/LFM2.5-1.2B-Instruct",
    "SmolLM3 3B (2025)": "HuggingFaceTB/SmolLM3-3B",
}

PRESETS = {
    "Balanced": SamplingParams(),
    "Creative": SamplingParams(temperature=1.4, top_k=100, top_p=0.95, repetition_penalty=1.0),
    "Focused": SamplingParams(temperature=0.5, top_k=20, top_p=0.6, repetition_penalty=1.2),
    "Deterministic (greedy)": SamplingParams(do_sample=False, repetition_penalty=1.0),
}

st.title("🎲 LLM Sampling Playground")
st.markdown(
    "Experiment with sampling parameters and watch how they reshape the model's "
    "token-by-token probability distribution."
)

st.sidebar.header("🎛️ Sampling Parameters")

preset = st.sidebar.selectbox("Preset", list(PRESETS))
base = PRESETS[preset]

do_sample = st.sidebar.toggle("Enable sampling", value=base.do_sample)
temperature = st.sidebar.slider(
    "🌡️ Temperature",
    0.1, 2.0, base.temperature if do_sample else 1.0, 0.05,
    help="Scales the logits before sampling. Higher = more random, lower = more focused.",
    disabled=not do_sample,
)
top_k = st.sidebar.slider(
    "🔝 Top-K", 1, 100, base.top_k, 1,
    help="Only sample from the K most probable tokens.",
    disabled=not do_sample,
)
top_p = st.sidebar.slider(
    "🎯 Top-P (nucleus)", 0.05, 1.0, base.top_p, 0.05,
    help="Only sample from the smallest set of tokens whose probabilities sum to P.",
    disabled=not do_sample,
)
repetition_penalty = st.sidebar.slider(
    "🔁 Repetition penalty", 1.0, 2.0, base.repetition_penalty, 0.05,
    help="Penalizes tokens that already appeared in the output.",
)
max_new_tokens = st.sidebar.slider("📏 Max new tokens", 10, 200, 60, 10)
seed = st.sidebar.number_input("Seed (0 = random each run)", 0, 2**31 - 1, base.seed or 0)

model_label = st.sidebar.selectbox("Model", list(MODELS), index=1)
model_name = MODELS[model_label]

prompt = st.text_area("📝 Your prompt", value="The meaning of life is", height=100)

if "usage_totals" not in st.session_state:
    st.session_state.usage_totals = {
        "generations": 0,
        "prompt_tokens": 0,
        "generated_tokens": 0,
    }

if "history" not in st.session_state:
    st.session_state.history = []

if st.button("🚀 Generate", type="primary"):
    with st.spinner("Loading model & generating…"):
        model, _ = load_model(model_name)
        params = SamplingParams(
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
            seed=seed or None,
        )
        result = generate(prompt, params, model_name)
        totals = st.session_state.usage_totals
        totals["generations"] += 1
        totals["prompt_tokens"] += result.usage.prompt_tokens
        totals["generated_tokens"] += result.usage.generated_tokens
        st.session_state.history.append(
            {
                "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "model": model_name,
                "prompt": prompt,
                "output": result.text,
                "temperature": params.temperature,
                "top_k": params.top_k,
                "top_p": params.top_p,
                "do_sample": params.do_sample,
                "prompt_tokens": result.usage.prompt_tokens,
                "generated_tokens": result.usage.generated_tokens,
            }
        )

    col_text, col_chart = st.columns([3, 2])

    with col_text:
        st.subheader("Generated output")
        st.markdown(f"> {prompt}**{result.text}**")
        mode = "sampling" if params.do_sample else "greedy"
        st.caption(
            f"Model: `{model_name}` · mode: {mode} · temp: {temperature} · "
            f"top_k: {top_k} · top_p: {top_p} · penalty: {repetition_penalty} · "
            f"seed: {params.seed or 'random'}"
        )
        usage = result.usage
        st.caption(
            f"🔢 Tokens this generation: {usage.prompt_tokens} prompt + "
            f"{usage.generated_tokens} generated = {usage.total_tokens}"
        )

    with col_chart:
        st.subheader("Token confidence")
        st.caption("Probability the model assigned to each generated token.")
        if result.probabilities:
            st.bar_chart(
                {
                    "token": [repr(t) for t in result.tokens],
                    "probability": result.probabilities,
                },
                x="token",
                y="probability",
                height=420,
            )
        else:
            st.info("No tokens were generated — increase max new tokens.")

st.sidebar.divider()
st.sidebar.subheader("📊 Session usage")
totals = st.session_state.usage_totals
col1, col2, col3 = st.sidebar.columns(3)
col1.metric("Runs", totals["generations"])
col2.metric("Prompt", totals["prompt_tokens"])
col3.metric("Generated", totals["generated_tokens"])
if totals["generations"] and st.sidebar.button("Reset"):
    st.session_state.usage_totals = {"generations": 0, "prompt_tokens": 0, "generated_tokens": 0}
    st.rerun()

with st.expander("⚔️ Active model comparison", expanded=False):
    st.caption(
        "Generate with the same prompt, seed and parameters on several models, "
        "then compare outputs, token consumption and speed."
    )
    default_compare = [label for label in MODELS if "GPT-2 (" in label or "LFM2.5 350M" in label]
    chosen = st.multiselect("Models to compare", list(MODELS), default=default_compare)

    if st.button("▶️ Run comparison", disabled=not chosen, type="primary"):
        comparison_params = SamplingParams(
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
            seed=seed or None,
        )
        rows = []
        for label in chosen:
            with st.spinner(f"Generating with {label}…"):
                start = time.perf_counter()
                result = generate(prompt, comparison_params, MODELS[label])
                elapsed = time.perf_counter() - start
            rows.append(
                comparison_summary_row(label, result, elapsed, max_new_tokens)
            )
            st.session_state.usage_totals["generations"] += 1
            st.session_state.usage_totals["prompt_tokens"] += result.usage.prompt_tokens
            st.session_state.usage_totals["generated_tokens"] += result.usage.generated_tokens
        st.session_state.last_comparison = rows

    if st.session_state.get("last_comparison"):
        rows = st.session_state.last_comparison
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.bar_chart(
            [{**r, "output": None} for r in rows],
            x="model",
            y=["prompt_tokens", "generated_tokens"],
            height=380,
        )
        st.caption(
            "`hit_max_tokens` = True means the run was cut off by your max-new-tokens "
            "limit — token counts are only comparable for runs that stopped on their own."
        )
        st.caption("Tokens per second is a rough CPU figure — larger models are slower.")

with st.expander("⚖️ Tokenizer comparison for this prompt", expanded=False):
    st.caption(
        "Same text, different tokenizers: how many tokens each model's tokenizer "
        "needs for your prompt. Loads tokenizers only — no model weights."
    )
    counts = compare_token_counts(
        prompt, {label: load_tokenizer(name) for label, name in MODELS.items()}
    )
    st.bar_chart(counts, x="model", y="prompt_tokens", height=380)

with st.expander("📜 Generation history", expanded=False):
    if st.session_state.history:
        st.dataframe(
            st.session_state.history,
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "⬇️ Export as CSV",
            history_to_csv(st.session_state.history),
            file_name="sampling-history.csv",
            mime="text/csv",
        )
        if st.button("🗑️ Clear history"):
            st.session_state.history = []
            st.rerun()
        if len({row["model"] for row in st.session_state.history}) > 1:
            st.subheader("Token usage by model (session average)")
            st.bar_chart(
                avg_tokens_by_model(st.session_state.history),
                y=["prompt_tokens", "generated_tokens"],
                height=380,
            )
    else:
        st.info("No generations yet — hit 🚀 Generate to start building history.")
