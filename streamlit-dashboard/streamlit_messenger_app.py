from __future__ import annotations

import html
import importlib.util
import json
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import streamlit as st


APP_DIR = Path(__file__).resolve().parent
SOURCE_DASHBOARD_DIR = APP_DIR.parent / "streamlit-dashboard"
DEFAULT_MODEL_DIR = APP_DIR.parent / "model" / "manuka-model-0527"
REQUIRED_PACKAGES = {
    "torch": "torch",
    "sentencepiece": "sentencepiece",
}
LFS_POINTER_PREFIX = "version https://git-lfs.github.com/spec/v1"

if str(SOURCE_DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DASHBOARD_DIR))


st.set_page_config(
    page_title="Manuka Messenger",
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
        --surface-strong: #e8e0d2;
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
        background-color: var(--canvas);
        background-image:
            linear-gradient(rgba(20, 20, 19, 0.018) 1px, transparent 1px),
            linear-gradient(90deg, rgba(20, 20, 19, 0.014) 1px, transparent 1px),
            repeating-linear-gradient(135deg, rgba(204, 120, 92, 0.018) 0 1px, transparent 1px 12px),
            radial-gradient(circle at 50% 34%, rgba(230, 223, 216, 0.46), transparent 32rem);
        background-size: 48px 48px, 48px 48px, 100% 100%, auto;
        color: var(--ink);
    }
    .block-container {
        max-width: 1080px;
        padding: 1.2rem 1.4rem 7.25rem;
    }
    [data-testid="stSidebar"],
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
    .topbar {
        align-items: center;
        display: flex;
        justify-content: space-between;
        margin: 0 auto 1.2rem;
        max-width: 920px;
    }
    .brand-lockup {
        align-items: center;
        display: flex;
        gap: 0.75rem;
    }
    .brand-mark {
        align-items: center;
        background: var(--ink);
        border-radius: 999px;
        color: var(--canvas);
        display: inline-flex;
        font-family: Georgia, Cambria, "Times New Roman", serif;
        font-size: 1.05rem;
        height: 2.35rem;
        justify-content: center;
        line-height: 1;
        width: 2.35rem;
    }
    .brand-name {
        color: var(--ink);
        font-size: 0.95rem;
        font-weight: 650;
        line-height: 1.15;
    }
    .brand-sub {
        color: var(--muted);
        font-size: 0.76rem;
        line-height: 1.2;
        margin-top: 0.14rem;
    }
    .status-pill {
        align-items: center;
        background: rgba(250, 249, 245, 0.78);
        border: 1px solid var(--hairline);
        border-radius: 999px;
        color: var(--muted);
        display: inline-flex;
        font-size: 0.78rem;
        gap: 0.45rem;
        padding: 0.42rem 0.72rem;
    }
    .status-pill::before {
        background: var(--success);
        border-radius: 999px;
        content: "";
        height: 0.46rem;
        width: 0.46rem;
    }
    .status-pill.warn::before {
        background: var(--warning);
    }
    .status-pill.bad::before {
        background: var(--error);
    }
    .st-key-settings_popover div[data-testid="stPopover"] button {
        background: var(--canvas);
        border: 1px solid var(--hairline);
        border-radius: 999px;
        color: var(--ink);
        font-size: 0.85rem;
        min-height: 2.25rem;
        padding: 0.45rem 0.8rem;
    }
    .st-key-settings_popover div[data-testid="stPopover"] button:hover {
        background: var(--surface-card);
        border-color: var(--hairline);
        color: var(--ink);
    }
    div[data-testid="stPopoverBody"] {
        background: var(--canvas);
        border: 1px solid var(--hairline);
        border-radius: 16px;
        box-shadow: 0 18px 42px rgba(20, 20, 19, 0.12);
        max-height: min(78vh, 680px);
        overflow-y: auto;
        width: min(520px, calc(100vw - 2rem));
    }
    .settings-title {
        color: var(--ink);
        font-family: Georgia, Cambria, "Times New Roman", serif;
        font-size: 1.35rem;
        font-weight: 400;
        letter-spacing: -0.01em;
        margin: 0 0 0.8rem;
    }
    .status-list {
        display: grid;
        gap: 0.45rem;
        margin: 0.75rem 0 1rem;
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
        font-size: 0.74rem;
        font-weight: 650;
        line-height: 1.2;
        text-transform: uppercase;
    }
    .status-value {
        color: var(--ink);
        font-size: 0.84rem;
        line-height: 1.25;
        overflow-wrap: anywhere;
    }
    .chat-frame {
        display: flex;
        flex-direction: column;
        justify-content: flex-end;
        margin: 0 auto;
        max-width: 920px;
        min-height: calc(100vh - 13rem);
        padding: 1rem 0 2.5rem;
    }
    .empty-card {
        align-self: center;
        background: rgba(250, 249, 245, 0.76);
        border: 1px solid var(--hairline);
        border-radius: 16px;
        margin-bottom: 12vh;
        max-width: 33rem;
        padding: 1.2rem 1.3rem;
        text-align: center;
    }
    .empty-kicker {
        color: var(--primary);
        font-size: 0.74rem;
        font-weight: 650;
        letter-spacing: 1.5px;
        margin-bottom: 0.55rem;
        text-transform: uppercase;
    }
    .empty-title {
        color: var(--ink);
        font-family: Georgia, Cambria, "Times New Roman", serif;
        font-size: clamp(1.8rem, 4vw, 2.65rem);
        font-weight: 400;
        letter-spacing: -0.02em;
        line-height: 1.08;
        margin: 0;
    }
    .message-row {
        display: flex;
        margin: 0.42rem 0;
        width: 100%;
    }
    .message-row.user {
        justify-content: flex-end;
    }
    .message-row.assistant {
        justify-content: flex-start;
    }
    .bubble {
        border: 1px solid var(--hairline);
        border-radius: 18px;
        box-shadow: 0 8px 22px rgba(20, 20, 19, 0.045);
        font-size: 0.98rem;
        line-height: 1.55;
        max-width: min(72%, 42rem);
        overflow-wrap: anywhere;
        padding: 0.78rem 0.95rem;
        white-space: pre-wrap;
    }
    .bubble.user {
        background: var(--primary);
        border-color: var(--primary);
        border-bottom-right-radius: 6px;
        color: #ffffff;
    }
    .bubble.assistant {
        background: rgba(250, 249, 245, 0.88);
        border-bottom-left-radius: 6px;
        color: var(--body);
    }
    .bubble.error {
        background: #fff2ef;
        border-color: rgba(198, 69, 69, 0.28);
        color: #8f2929;
    }
    .bubble-meta {
        border-top: 1px solid var(--hairline);
        display: flex;
        flex-wrap: wrap;
        gap: 0.42rem;
        margin-top: 0.7rem;
        padding-top: 0.65rem;
    }
    .meta-chip {
        background: var(--surface-soft);
        border: 1px solid var(--hairline);
        border-radius: 999px;
        color: var(--muted);
        font-size: 0.72rem;
        padding: 0.22rem 0.5rem;
    }
    [data-testid="stChatInput"] {
        max-width: 920px;
        margin: 0 auto;
    }
    [data-testid="stChatInput"] textarea {
        background: #ffffff !important;
        border: 1px solid var(--hairline) !important;
        border-radius: 999px !important;
        box-shadow: 0 14px 38px rgba(20, 20, 19, 0.08) !important;
        color: var(--ink) !important;
        min-height: 3.25rem !important;
        padding: 0.88rem 1.2rem !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: var(--primary) !important;
        box-shadow:
            0 0 0 3px rgba(204, 120, 92, 0.14),
            0 14px 38px rgba(20, 20, 19, 0.08) !important;
    }
    .stTextInput input,
    .stNumberInput input,
    .stSelectbox div[data-baseweb="select"] > div {
        background: var(--canvas);
        border-color: var(--hairline);
        border-radius: 8px;
        color: var(--ink);
    }
    @media (max-width: 760px) {
        .block-container {
            padding-left: 0.9rem;
            padding-right: 0.9rem;
        }
        .topbar {
            align-items: flex-start;
            gap: 0.75rem;
        }
        .status-pill {
            display: none;
        }
        .chat-frame {
            min-height: calc(100vh - 12rem);
        }
        .bubble {
            max-width: 86%;
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
        return "model.pt is a Git LFS pointer outside this repository."

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
        return "model.pt is still a Git LFS pointer after `git lfs pull`."
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


def escape_message(value: str) -> str:
    return html.escape(value or "").replace("\n", "<br>")


def render_messages(messages: list[dict[str, Any]]) -> None:
    if not messages:
        st.markdown(
            """
            <section class="chat-frame">
                <div class="empty-card">
                    <div class="empty-kicker">Manuka local checkpoint</div>
                    <h1 class="empty-title">Messenger workspace</h1>
                </div>
            </section>
            """,
            unsafe_allow_html=True,
        )
        return

    rows = []
    for message in messages:
        role = "user" if message.get("role") == "user" else "assistant"
        error_class = " error" if message.get("error") else ""
        body = escape_message(str(message.get("content", "")))
        meta = message.get("meta") or {}
        meta_html = ""
        if meta:
            chips = "".join(
                f'<span class="meta-chip">{html.escape(label)}: {html.escape(str(value))}</span>'
                for label, value in meta.items()
            )
            meta_html = f'<div class="bubble-meta">{chips}</div>'
        rows.append(
            f'<div class="message-row {role}">'
            f'<div class="bubble {role}{error_class}">{body}{meta_html}</div>'
            f'</div>'
        )
    st.markdown(f'<section class="chat-frame">{"".join(rows)}</section>', unsafe_allow_html=True)


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


def append_assistant(content: str, *, meta: dict[str, Any] | None = None, error: bool = False) -> None:
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": content,
            "meta": meta or {},
            "error": error,
        }
    )


if "messages" not in st.session_state:
    st.session_state.messages = []

with st.container(key="settings_popover"):
    left, right = st.columns([0.78, 0.22], vertical_alignment="center")
    with left:
        st.markdown(
            """
            <div class="topbar">
                <div class="brand-lockup">
                    <span class="brand-mark">M</span>
                    <span>
                        <div class="brand-name">Manuka Messenger</div>
                        <div class="brand-sub">Local checkpoint chat</div>
                    </span>
                </div>
                <span class="status-pill">ready</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        with st.popover("Settings", use_container_width=True):
            st.markdown('<div class="settings-title">Model settings</div>', unsafe_allow_html=True)
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

render_messages(st.session_state.messages)

prompt = st.chat_input("Message Manuka")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    if cfg_error:
        append_assistant(f"Could not read config.json: {cfg_error}", error=True)
    elif lfs_error:
        append_assistant(lfs_error, error=True)
    elif not files_ok:
        append_assistant("Model files are incomplete.", error=True)
    elif missing:
        append_assistant(
            "Runtime packages are missing: " + ", ".join(missing),
            error=True,
        )
    else:
        try:
            model_stamp, config_stamp = cache_stamps(model_dir)
            with st.spinner("Generating response"):
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
            append_assistant(
                result.text,
                meta={
                    "tokens": f"{result.token_count:,}",
                    "elapsed": f"{result.elapsed_seconds:.2f}s",
                    "device": result.device,
                    "params": f"{info['parameters']:,}",
                },
            )
        except Exception as exc:
            append_assistant(f"Inference failed: {exc}", error=True)
            st.session_state.last_traceback = traceback.format_exc()
    st.rerun()
