from __future__ import annotations

import html
import importlib.util
import json
import traceback
from pathlib import Path
from typing import Any

import streamlit as st


APP_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = APP_DIR.parent / "model" / "manuka-model-0527"
REQUIRED_PACKAGES = {
    "torch": "torch",
    "sentencepiece": "sentencepiece",
}


st.set_page_config(
    page_title="Inference Demo",
    page_icon="M",
    layout="wide",
)


st.markdown(
    """
    <style>
    .stApp {
        background:
            linear-gradient(180deg, #fbfaf6 0%, #eef5f2 52%, #f5f1e8 100%);
        color: #181917;
    }
    .block-container {
        max-width: 1120px;
        padding-top: 2.25rem;
        padding-bottom: 4rem;
    }
    [data-testid="stSidebar"] {
        background: #171916;
        color: #f4efe4;
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] p {
        color: #f4efe4;
    }
    h1, h2, h3 {
        letter-spacing: 0;
    }
    .manuka-title {
        border-top: 5px solid #181917;
        border-bottom: 1px solid #cfc6b6;
        padding: 1.35rem 0 1.1rem;
        margin-bottom: 1.4rem;
    }
    .manuka-title .eyebrow {
        color: #0f766e;
        font-size: 0.84rem;
        font-weight: 800;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }
    .manuka-title h1 {
        color: #181917;
        font-family: Georgia, Cambria, "Times New Roman", serif;
        font-size: clamp(2.2rem, 4.5vw, 4.9rem);
        line-height: 0.96;
        margin: 0;
        font-weight: 700;
    }
    .manuka-title p {
        color: #4d5149;
        font-size: 1rem;
        max-width: 760px;
        margin: 0.85rem 0 0;
    }
    .status-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.75rem;
        margin: 1rem 0 1.45rem;
    }
    .status-card {
        background: rgba(255, 255, 255, 0.7);
        border: 1px solid #d8d0c1;
        border-radius: 8px;
        padding: 0.85rem 0.9rem;
        min-height: 86px;
    }
    .status-card strong {
        display: block;
        color: #606458;
        font-size: 0.74rem;
        margin-bottom: 0.38rem;
        text-transform: uppercase;
    }
    .status-card span {
        color: #171916;
        font-size: 0.98rem;
        font-weight: 750;
        overflow-wrap: anywhere;
    }
    .status-ok {
        border-top: 4px solid #0f766e;
    }
    .status-warn {
        border-top: 4px solid #b45309;
    }
    .status-bad {
        border-top: 4px solid #b91c1c;
    }
    .response-box {
        background: #111410;
        color: #f8f1e6;
        border: 1px solid #2d332a;
        border-radius: 8px;
        padding: 1rem 1.05rem;
        min-height: 140px;
        white-space: pre-wrap;
        overflow-wrap: anywhere;
        font-size: 1.04rem;
        line-height: 1.6;
    }
    .small-note {
        color: #5f645b;
        font-size: 0.88rem;
        line-height: 1.45;
    }
    @media (max-width: 850px) {
        .status-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
        .manuka-title h1 {
            font-size: 2.4rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def format_bytes(value: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} {unit}"
        size /= 1024
    return f"{value} B"


def load_config(model_dir: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        with open(model_dir / "config.json", encoding="utf-8") as handle:
            return json.load(handle), None
    except Exception as exc:
        return None, str(exc)


def missing_dependencies() -> list[str]:
    return [package for package, module in REQUIRED_PACKAGES.items() if importlib.util.find_spec(module) is None]


def model_file_state(model_dir: Path, cfg: dict[str, Any] | None) -> tuple[list[dict[str, str]], bool]:
    checks: list[dict[str, str]] = []
    ok = True

    def add(label: str, value: str, state: str) -> None:
        nonlocal ok
        checks.append({"label": label, "value": value, "state": state})
        if state == "bad":
            ok = False

    if not model_dir.exists():
        add("Model Dir", "missing", "bad")
        add("Config", "not checked", "bad")
        add("Weights", "not checked", "bad")
        add("Tokenizer", "not checked", "bad")
        return checks, False

    add("Model Dir", str(model_dir), "ok")

    config_path = model_dir / "config.json"
    add("Config", "found" if config_path.exists() else "missing", "ok" if config_path.exists() else "bad")

    model_path = model_dir / "model.pt"
    if model_path.exists():
        add("Weights", format_bytes(model_path.stat().st_size), "ok")
    else:
        add("Weights", "missing", "bad")

    tokenizer_name = (cfg or {}).get("tokenizer_model", "spm16k.model")
    tokenizer_path = model_dir / tokenizer_name
    add("Tokenizer", tokenizer_name if tokenizer_path.exists() else "missing", "ok" if tokenizer_path.exists() else "bad")
    return checks, ok


def render_status_cards(checks: list[dict[str, str]]) -> None:
    cards = []
    for check in checks:
        label = html.escape(check["label"])
        value = html.escape(check["value"])
        state = html.escape(check["state"])
        cards.append(
            f"""
            <div class="status-card status-{state}">
                <strong>{label}</strong>
                <span>{value}</span>
            </div>
            """
        )
    st.markdown(f"<div class='status-grid'>{''.join(cards)}</div>", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_generator(model_dir: str, device: str, model_stamp: int, config_stamp: int):
    del model_stamp, config_stamp
    from manuka_inference import ManukaGenerator

    requested_device = None if device == "auto" else device
    return ManukaGenerator(model_dir, device=requested_device)


def cache_stamps(model_dir: Path) -> tuple[int, int]:
    model_path = model_dir / "model.pt"
    config_path = model_dir / "config.json"
    model_stamp = model_path.stat().st_mtime_ns if model_path.exists() else 0
    config_stamp = config_path.stat().st_mtime_ns if config_path.exists() else 0
    return model_stamp, config_stamp


st.markdown(
    """
    <div class="manuka-title">
        <div class="eyebrow">Local checkpoint</div>
        <h1>Inference Demo</h1>
    </div>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.header("Runtime")
    model_dir_text = st.text_input("Model directory", value=str(DEFAULT_MODEL_DIR))
    model_dir = Path(model_dir_text).expanduser()
    cfg, cfg_error = load_config(model_dir)

    max_len = int((cfg or {}).get("max_len", 512))
    gen_cfg = (cfg or {}).get("generation", {})
    device_choice = st.selectbox("Device", ["auto", "cpu", "cuda"], index=0)
    max_new_limit = max(1, min(256, max_len - 3))
    default_max_new = min(int(gen_cfg.get("max_new_tokens", 96)), max_new_limit)
    max_new_tokens = st.slider("Max new tokens", 1, max_new_limit, default_max_new)
    min_new_tokens = st.slider("Min new tokens", 0, min(48, max_new_tokens), min(12, max_new_tokens))
    temperature = st.slider("Temperature", 0.1, 1.5, float(gen_cfg.get("temperature", 0.72)), 0.01)
    top_p = st.slider("Top p", 0.05, 1.0, float(gen_cfg.get("top_p", 0.88)), 0.01)
    top_k = st.slider("Top k", 0, 200, int(gen_cfg.get("top_k", 60)))
    repetition_penalty = st.slider(
        "Repetition penalty",
        1.0,
        1.5,
        float(gen_cfg.get("repetition_penalty", 1.12)),
        0.01,
    )
    no_repeat_ngram_size = st.slider("No repeat ngram", 0, 8, int(gen_cfg.get("no_repeat_ngram_size", 4)))
    use_seed = st.checkbox("Use seed", value=False)
    seed = st.number_input("Seed", min_value=0, max_value=2**31 - 1, value=42, disabled=not use_seed)


if cfg_error:
    st.error(f"Could not read config.json: {cfg_error}")

checks, files_ok = model_file_state(model_dir, cfg)
missing = missing_dependencies()
dep_state = "ok" if not missing else "warn"
checks.append(
    {
        "label": "Dependencies",
        "value": "ready" if not missing else ", ".join(missing),
        "state": dep_state,
    }
)
render_status_cards(checks)

if missing:
    st.warning(
        "Missing runtime packages. Install the packages in streamlit_requirements.txt before loading the model."
    )

prompt_default = ""
prompt = st.text_area("Prompt", value=prompt_default, height=145)
generate_clicked = st.button("Generate", type="primary", use_container_width=True)

if "runs" not in st.session_state:
    st.session_state.runs = []

if generate_clicked:
    if not prompt.strip():
        st.error("Prompt is empty.")
    elif not files_ok:
        st.error("Model files are incomplete.")
    elif missing:
        st.error("Cannot run inference until the missing packages are installed.")
    else:
        try:
            model_stamp, config_stamp = cache_stamps(model_dir)
            with st.spinner("Loading checkpoint and generating response"):
                generator = get_generator(str(model_dir), device_choice, model_stamp, config_stamp)
                result = generator.generate_reply(
                    prompt,
                    max_ctx=max_len,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    min_new_tokens=min_new_tokens,
                    seed=int(seed) if use_seed else None,
                )
                info = generator.info()
            st.session_state.runs.append(
                {
                    "prompt": prompt,
                    "response": result.text,
                    "tokens": result.token_count,
                    "prompt_tokens": result.prompt_token_count,
                    "elapsed": result.elapsed_seconds,
                    "device": result.device,
                    "parameters": info["parameters"],
                }
            )
        except Exception as exc:
            st.error(f"Inference failed: {exc}")
            with st.expander("Traceback"):
                st.code(traceback.format_exc())

latest = st.session_state.runs[-1] if st.session_state.runs else None
left, right = st.columns([2.2, 1])
with left:
    st.subheader("Response")
    response_text = latest["response"] if latest else ""
    st.markdown(
        f"<div class='response-box'>{html.escape(response_text) if response_text else '&nbsp;'}</div>",
        unsafe_allow_html=True,
    )

with right:
    st.subheader("Run stats")
    if latest:
        st.metric("Generated tokens", latest["tokens"])
        st.metric("Prompt tokens", latest["prompt_tokens"])
        st.metric("Elapsed", f"{latest['elapsed']:.2f}s")
        st.metric("Device", latest["device"])
        st.caption(f"Parameters: {latest['parameters']:,}")
    else:
        st.markdown("<p class='small-note'>No generation has run in this Streamlit session.</p>", unsafe_allow_html=True)

if st.session_state.runs:
    st.subheader("Recent runs")
    for index, run in enumerate(reversed(st.session_state.runs[-5:]), start=1):
        with st.expander(f"Run {len(st.session_state.runs) - index + 1}: {run['tokens']} tokens"):
            st.markdown("Prompt")
            st.code(run["prompt"])
            st.markdown("Response")
            st.write(run["response"])
