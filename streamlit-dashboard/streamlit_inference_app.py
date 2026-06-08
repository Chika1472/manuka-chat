from __future__ import annotations

import html
import importlib.util
import json
import subprocess
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
LFS_POINTER_PREFIX = "version https://git-lfs.github.com/spec/v1"


st.set_page_config(
    page_title="Inference Demo",
    page_icon="M",
    layout="wide",
)


st.markdown(
    """
    <style>
    :root {
        --canvas: #faf9f5;
        --surface-soft: #f5f0e8;
        --surface-card: #efe9de;
        --surface-dark: #181715;
        --surface-dark-elevated: #252320;
        --ink: #141413;
        --body: #3d3d3a;
        --muted: #6c6a64;
        --muted-soft: #8e8b82;
        --hairline: #e6dfd8;
        --primary: #cc785c;
        --primary-active: #a9583e;
        --success: #5db872;
        --warning: #d4a017;
        --error: #c64545;
    }
    .stApp {
        background:
            radial-gradient(circle at 50% 45%, rgba(230, 223, 216, 0.46), transparent 34rem),
            var(--canvas);
        color: var(--ink);
    }
    .block-container {
        max-width: 980px;
        padding-top: 1.25rem;
        padding-bottom: 4rem;
    }
    [data-testid="stSidebar"] {
        display: none;
    }
    [data-testid="collapsedControl"],
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    [data-testid="stStatusWidget"],
    .stDeployButton,
    #MainMenu,
    footer {
        display: none !important;
    }
    header[data-testid="stHeader"] {
        background: transparent;
    }
    h1, h2, h3 {
        letter-spacing: 0;
        color: var(--ink);
    }
    .manuka-hero {
        text-align: center;
        transition: margin 160ms ease;
    }
    .manuka-hero.idle {
        margin-top: clamp(9rem, 28vh, 18rem);
    }
    .manuka-hero.active {
        margin-top: 3.5rem;
    }
    .manuka-hero .eyebrow {
        color: var(--primary);
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 1.5px;
        line-height: 1.4;
        margin-bottom: 0.7rem;
        text-transform: uppercase;
    }
    .manuka-hero h1 {
        color: var(--ink);
        font-family: Georgia, Cambria, "Times New Roman", serif;
        font-size: clamp(1.8rem, 2.6vw, 2.35rem);
        font-weight: 400;
        letter-spacing: -0.02em;
        line-height: 1.12;
        margin: 0;
    }
    .st-key-composer_shell {
        background: var(--canvas);
        border: 1px solid var(--hairline);
        border-radius: 999px;
        box-shadow: 0 18px 46px rgba(20, 20, 19, 0.08);
        margin: 1.7rem auto 0;
        max-width: 780px;
        padding: 0.32rem;
    }
    .st-key-composer_shell [data-testid="stHorizontalBlock"] {
        align-items: center;
        gap: 0.35rem;
    }
    .st-key-composer_shell [data-testid="column"] {
        padding: 0 !important;
    }
    .st-key-composer_shell div[data-testid="stPopover"] button {
        width: 46px;
        height: 46px;
        min-height: 46px;
        border-radius: 999px;
        border: 0;
        background: var(--surface-soft);
        color: var(--ink);
        box-shadow: none;
        font-size: 22px;
        font-weight: 300;
        padding: 0;
    }
    .st-key-composer_shell div[data-testid="stPopover"] button svg {
        display: none;
    }
    .st-key-composer_shell div[data-testid="stPopover"] button:hover {
        color: var(--ink);
        background: var(--surface-card);
    }
    div[data-testid="stPopoverBody"] {
        background: var(--canvas);
        border: 1px solid var(--hairline);
        border-radius: 16px;
        max-height: min(72vh, 620px);
        overflow-y: auto;
        box-shadow: 0 18px 42px rgba(20, 20, 19, 0.12);
        width: min(520px, calc(100vw - 2rem));
    }
    .st-key-composer_shell .stTextArea textarea {
        min-height: 46px !important;
        height: 46px !important;
        resize: none;
        border: 0;
        border-radius: 999px;
        background: transparent;
        box-shadow: none;
        color: var(--ink);
        font-size: 16px;
        line-height: 1.35;
        padding: 13px 18px;
    }
    .st-key-composer_shell .stTextArea textarea:focus {
        border: 0;
        box-shadow: none;
    }
    .stTextInput input,
    .stNumberInput input,
    .stSelectbox div[data-baseweb="select"] > div {
        border-color: var(--hairline);
        border-radius: 8px;
        background: var(--canvas);
        color: var(--ink);
    }
    .st-key-composer_shell [data-testid="baseButton-primary"] {
        width: 46px;
        min-width: 46px;
        height: 46px;
        min-height: 46px;
        border: 0;
        border-radius: 999px;
        background: var(--primary);
        color: #ffffff;
        box-shadow: none;
        font-size: 20px;
        font-weight: 600;
        padding: 0;
    }
    .st-key-composer_shell [data-testid="baseButton-primary"]:hover,
    .st-key-composer_shell [data-testid="baseButton-primary"]:focus {
        background: var(--primary-active);
        color: #ffffff;
        border: 0;
    }
    .settings-title {
        color: var(--ink);
        font-family: Georgia, Cambria, "Times New Roman", serif;
        font-size: 1.35rem;
        font-weight: 400;
        letter-spacing: -0.01em;
        margin: 0 0 0.2rem;
    }
    .status-list {
        display: grid;
        gap: 0.45rem;
        margin: 0.35rem 0 1rem;
    }
    .status-row {
        align-items: center;
        background: var(--surface-soft);
        border: 1px solid var(--hairline);
        border-radius: 8px;
        display: grid;
        gap: 0.65rem;
        grid-template-columns: 0.55rem 6.8rem minmax(0, 1fr);
        padding: 0.55rem 0.65rem;
    }
    .status-dot {
        border-radius: 999px;
        height: 0.45rem;
        width: 0.45rem;
    }
    .status-ok .status-dot {
        background: var(--success);
    }
    .status-warn .status-dot {
        background: var(--warning);
    }
    .status-bad .status-dot {
        background: var(--error);
    }
    .status-label {
        color: var(--muted);
        font-size: 0.76rem;
        font-weight: 600;
        line-height: 1.2;
        text-transform: uppercase;
    }
    .status-value {
        color: var(--ink);
        font-size: 0.84rem;
        line-height: 1.25;
        overflow-wrap: anywhere;
    }
    .alert-copy {
        color: var(--muted);
        font-size: 0.9rem;
        margin: 0.5rem 0 0;
        text-align: center;
    }
    .result-card {
        background: var(--surface-card);
        border: 1px solid var(--hairline);
        border-radius: 12px;
        color: var(--ink);
        margin: 2rem auto 0;
        max-width: 860px;
        padding: clamp(1.2rem, 3vw, 2rem);
    }
    .result-kicker {
        color: var(--primary);
        font-size: 0.74rem;
        font-weight: 600;
        letter-spacing: 1.5px;
        margin-bottom: 0.8rem;
        text-transform: uppercase;
    }
    .response-copy {
        color: var(--body);
        font-size: 1.03rem;
        line-height: 1.65;
        min-height: 5rem;
        white-space: pre-wrap;
        overflow-wrap: anywhere;
    }
    .stats-strip {
        border-top: 1px solid var(--hairline);
        display: grid;
        gap: 0.75rem;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        margin-top: 1.45rem;
        padding-top: 1rem;
    }
    .stat-item {
        background: var(--canvas);
        border: 1px solid var(--hairline);
        border-radius: 8px;
        padding: 0.72rem 0.8rem;
        min-width: 0;
    }
    .stat-label {
        color: var(--muted);
        display: block;
        font-size: 0.72rem;
        line-height: 1.3;
        margin-bottom: 0.25rem;
    }
    .stat-value {
        color: var(--ink);
        display: block;
        font-size: 0.95rem;
        font-weight: 600;
        line-height: 1.35;
        overflow-wrap: anywhere;
    }
    [data-testid="stExpander"] {
        background: rgba(239, 233, 222, 0.72);
        border-color: var(--hairline);
        border-radius: 8px;
    }
    .small-note {
        color: var(--muted);
        font-size: 0.88rem;
        line-height: 1.45;
    }
    @media (max-width: 850px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
        .manuka-hero.idle {
            margin-top: 7rem;
        }
        .stats-strip {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
        .st-key-composer_shell {
            border-radius: 24px;
            max-width: 100%;
        }
        .st-key-composer_shell div[data-testid="stPopover"] button,
        .st-key-composer_shell [data-testid="baseButton-primary"] {
            min-height: 42px;
            height: 42px;
            width: 42px;
        }
        .st-key-composer_shell .stTextArea textarea {
            min-height: 42px !important;
            height: 42px !important;
            padding-top: 11px;
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


def is_lfs_pointer(path: Path) -> bool:
    if not path.exists() or path.stat().st_size > 1024:
        return False
    try:
        return path.read_text(encoding="utf-8", errors="ignore").startswith(LFS_POINTER_PREFIX)
    except OSError:
        return False


def materialize_lfs_checkpoint(model_dir: Path) -> str | None:
    model_path = model_dir / "model.pt"
    if not is_lfs_pointer(model_path):
        return None

    repo_dir = APP_DIR.parent
    try:
        include_path = model_path.resolve().relative_to(repo_dir.resolve()).as_posix()
    except ValueError:
        return "model.pt is a Git LFS pointer outside this repository, so the app cannot download it automatically."

    try:
        subprocess.run(
            ["git", "lfs", "pull", "--include", include_path],
            cwd=repo_dir,
            check=True,
            capture_output=True,
            text=True,
            timeout=180,
        )
    except Exception as exc:
        return f"model.pt is a Git LFS pointer, and `git lfs pull` failed: {exc}"

    if is_lfs_pointer(model_path):
        return "model.pt is still a Git LFS pointer after `git lfs pull`; the real checkpoint was not downloaded."
    return None


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
    if is_lfs_pointer(model_path):
        add("Weights", "Git LFS pointer", "bad")
    elif model_path.exists():
        add("Weights", format_bytes(model_path.stat().st_size), "ok")
    else:
        add("Weights", "missing", "bad")

    tokenizer_name = (cfg or {}).get("tokenizer_model", "spm16k.model")
    tokenizer_path = model_dir / tokenizer_name
    add("Tokenizer", tokenizer_name if tokenizer_path.exists() else "missing", "ok" if tokenizer_path.exists() else "bad")
    return checks, ok


def render_status_list(checks: list[dict[str, str]]) -> None:
    rows = []
    for check in checks:
        state = html.escape(check["state"])
        label = html.escape(check["label"])
        value = html.escape(check["value"])
        rows.append(
            f'<div class="status-row status-{state}">'
            f'<span class="status-dot"></span>'
            f'<span class="status-label">{label}</span>'
            f'<span class="status-value">{value}</span>'
            f'</div>'
        )
    st.markdown(f"<div class='status-list'>{''.join(rows)}</div>", unsafe_allow_html=True)


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


def render_latest_run(run: dict[str, Any]) -> None:
    response = html.escape(str(run["response"])) or "&nbsp;"
    stats = [
        ("Generated", f"{run['tokens']:,}"),
        ("Prompt", f"{run['prompt_tokens']:,}"),
        ("Elapsed", f"{run['elapsed']:.2f}s"),
        ("Device", str(run["device"])),
        ("Parameters", f"{run['parameters']:,}"),
    ]
    stat_html = "".join(
        f"""
        <div class="stat-item">
            <span class="stat-label">{html.escape(label)}</span>
            <span class="stat-value">{html.escape(value)}</span>
        </div>
        """
        for label, value in stats
    )
    st.markdown(
        f"""
        <section class="result-card">
            <div class="result-kicker">Response</div>
            <div class="response-copy">{response}</div>
            <div class="stats-strip">{stat_html}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


if "runs" not in st.session_state:
    st.session_state.runs = []

latest = st.session_state.runs[-1] if st.session_state.runs else None
hero_state = "active" if latest else "idle"
st.markdown(
    f"""
    <section class="manuka-hero {hero_state}">
        <div class="eyebrow">Manuka local checkpoint</div>
        <h1>대화를 시작해 보세요.</h1>
    </section>
    """,
    unsafe_allow_html=True,
)

with st.container(key="composer_shell"):
    settings_col, prompt_col, run_col = st.columns([0.48, 6.5, 0.48], gap="small")
    with settings_col:
        with st.popover("+", use_container_width=True):
            st.markdown(
                """
                <div class="settings-title">Model settings</div>
                """,
                unsafe_allow_html=True,
            )
            model_dir_text = st.text_input("Model directory", value=str(DEFAULT_MODEL_DIR))
            model_dir = Path(model_dir_text).expanduser()
            lfs_error = materialize_lfs_checkpoint(model_dir)
            cfg, cfg_error = load_config(model_dir)

            max_len = max(4, int((cfg or {}).get("max_len", 512)))
            gen_cfg = (cfg or {}).get("generation", {})
            device_choice = st.selectbox("Device", ["auto", "cpu", "cuda"], index=0)
            context_window = st.slider("Context length", 4, max_len, max_len)
            max_new_limit = max(1, min(256, context_window - 3))
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

            checks, files_ok = model_file_state(model_dir, cfg)
            missing = missing_dependencies()
            checks.append(
                {
                    "label": "Dependencies",
                    "value": "ready" if not missing else ", ".join(missing),
                    "state": "ok" if not missing else "warn",
                }
            )
            render_status_list(checks)

    with prompt_col:
        prompt = st.text_area("Prompt", value="", placeholder="무엇이든 물어보세요", height=46, label_visibility="collapsed")

    with run_col:
        generate_clicked = st.button("→", type="primary", use_container_width=True)

if cfg_error:
    st.error(f"Could not read config.json: {cfg_error}")
if lfs_error:
    st.error(lfs_error)
if missing:
    st.warning("Missing runtime packages. Install the packages in requirements.txt before loading the model.")

if generate_clicked:
    if not prompt.strip():
        st.error("Prompt is empty.")
    elif lfs_error:
        st.error("The real model checkpoint has not been downloaded from Git LFS.")
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
                    max_ctx=context_window,
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
if latest:
    render_latest_run(latest)

if st.session_state.runs:
    with st.expander("Chat History", expanded=False):
        for index, run in enumerate(reversed(st.session_state.runs[-5:]), start=1):
            run_number = len(st.session_state.runs) - index + 1
            st.markdown(f"**Run {run_number}: {run['tokens']} tokens**")
            st.markdown("Prompt")
            st.code(run["prompt"])
            st.markdown("Response")
            st.write(run["response"])
            if index < min(5, len(st.session_state.runs)):
                st.divider()
