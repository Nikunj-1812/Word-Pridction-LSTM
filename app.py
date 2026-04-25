import pickle
import importlib
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import streamlit as st


MODEL_PATH = Path("lstm_model.h5")
TOKENIZER_PATH = Path("tokenizer.pkl")
MAX_LEN_PATH = Path("max_len.pkl")


st.set_page_config(
    page_title="Next Word Prediction using LSTM",
    page_icon="assets/icon.png" if Path("assets/icon.png").exists() else None,
    layout="wide",
    initial_sidebar_state="collapsed",
)


def apply_styles() -> None:
    """Inject custom CSS for a polished SaaS-like interface."""
    st.markdown(
        """
        <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,500,0,0" />
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap" rel="stylesheet">
        <style>
            :root {
                --bg-main: radial-gradient(circle at 10% 10%, #eff4ff 0%, #f7f9fc 45%, #fbfcfe 100%);
                --card-bg: rgba(255, 255, 255, 0.92);
                --card-border: rgba(15, 23, 42, 0.09);
                --text-main: #111827;
                --text-muted: #64748b;
                --accent: #1d4ed8;
                --accent-strong: #1e40af;
                --accent-soft: rgba(29, 78, 216, 0.12);
                --shadow-soft: 0 12px 28px rgba(15, 23, 42, 0.08);
            }

            .stApp {
                background: var(--bg-main);
                color: var(--text-main);
                font-family: "Manrope", sans-serif;
            }

            .block-container {
                max-width: 980px;
                padding-top: 1rem !important;
                padding-bottom: 1rem;
            }

            .hero-card {
                background: var(--card-bg);
                backdrop-filter: blur(6px);
                border: 1px solid var(--card-border);
                border-radius: 14px;
                padding: 1.1rem 1.2rem 1rem 1.2rem;
                box-shadow: var(--shadow-soft);
                margin-bottom: 1rem;
            }

            .hero-title-row {
                display: flex;
                align-items: center;
                gap: 0.55rem;
                margin-bottom: 0.1rem;
            }

            .hero-title {
                font-size: 1.75rem;
                font-weight: 800;
                letter-spacing: -0.02em;
                color: var(--text-main);
                margin: 0;
            }

            .hero-subtitle {
                color: var(--text-muted);
                font-size: 0.96rem;
                margin: 0.15rem 0 0;
                font-weight: 500;
            }

            .hero-divider {
                margin-top: 0.85rem;
                height: 1px;
                width: 100%;
                background: linear-gradient(90deg, rgba(29, 78, 216, 0.35) 0%, rgba(148, 163, 184, 0.25) 100%);
            }

            .mat-icon {
                font-family: "Material Symbols Outlined";
                font-variation-settings: "FILL" 0, "wght" 500, "GRAD" 0, "opsz" 24;
                line-height: 1;
                vertical-align: middle;
                font-size: 1.2rem;
            }

            .content-shell {
                width: min(100%, 42rem);
                margin: 0 auto;
                padding: 0 0.4rem;
            }

            .section-label {
                display: inline-flex;
                align-items: center;
                gap: 0.45rem;
                font-weight: 700;
                color: #0f172a;
                margin-bottom: 0.55rem;
                font-size: 0.95rem;
                width: 100%;
            }

            .section-note {
                color: var(--text-muted);
                margin-top: -0.15rem;
                margin-bottom: 0.65rem;
                font-size: 0.88rem;
            }

            div.element-container {
                margin-bottom: 0.1rem !important;
            }

            .result-card {
                background: linear-gradient(135deg, #eef4ff 0%, #f2f7ff 55%, #f8fbff 100%);
                border: 1px solid rgba(29, 78, 216, 0.25);
                border-radius: 14px;
                padding: 0.95rem 1rem;
                margin-top: 0.75rem;
                animation: fadeInUp 0.35s ease;
            }

            .result-title {
                display: flex;
                align-items: center;
                gap: 0.45rem;
                font-size: 1rem;
                color: #0f172a;
                font-weight: 700;
                margin-bottom: 0.35rem;
            }

            .result-text {
                color: #0f172a;
                margin: 0;
                font-size: 1rem;
                line-height: 1.65;
            }

            .generated-word {
                background: var(--accent-soft);
                color: var(--accent-strong);
                font-weight: 700;
                padding: 0.15rem 0.42rem;
                border-radius: 8px;
                border: 1px solid rgba(15, 118, 110, 0.25);
                display: inline-block;
                margin: 0 0.12rem;
            }

            div[data-testid="stTextInput"] input {
                border-radius: 0 !important;
                border: 0 !important;
                height: 54px;
                min-height: 54px;
                padding: 0 1rem !important;
                font-size: 1rem;
                line-height: normal;
                box-sizing: border-box;
                background: transparent;
                transition: border-color 0.2s ease, box-shadow 0.2s ease;
            }

            div[data-testid="stTextInput"] {
                margin: 0 auto 0.35rem auto;
                width: 100%;
                max-width: 42rem;
            }

            div[data-testid="stTextInput"] > div {
                width: 100%;
                border-radius: 14px;
                overflow: hidden;
                background: #ffffff;
                border: 1px solid rgba(15, 23, 42, 0.18);
                box-sizing: border-box;
            }

            div[data-testid="stTextInput"] > div:focus-within {
                border-color: rgba(15, 23, 42, 0.18);
                box-shadow: none;
            }

            div[data-testid="stTextInput"] input::placeholder {
                color: #94a3b8;
                opacity: 1;
            }

            div[data-testid="stTextInput"] input:focus {
                outline: none;
                box-shadow: none;
            }

            div[data-testid="InputInstructions"] {
                display: none !important;
                visibility: hidden !important;
                height: 0 !important;
                margin: 0 !important;
                padding: 0 !important;
            }

            div[data-testid="stSlider"] {
                margin-top: 0.05rem;
                margin-bottom: 0.3rem;
                padding-right: 0;
                max-width: 42rem;
                margin-left: auto;
                margin-right: auto;
            }

            div[data-testid="stSlider"] > div {
                padding-top: 0.1rem;
            }

            div[data-testid="stSlider"] label {
                margin-bottom: 0.45rem;
            }

            div.stButton > button {
                width: 100%;
                border-radius: 12px;
                min-height: 48px;
                border: 1px solid transparent;
                background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%);
                color: #ffffff;
                font-weight: 700;
                letter-spacing: 0.01em;
                box-shadow: 0 8px 20px rgba(29, 78, 216, 0.3);
                transition: transform 0.18s ease, box-shadow 0.18s ease, filter 0.18s ease;
            }

            div[data-testid="stButton"] {
                max-width: 42rem;
                margin: 0.1rem auto 0 auto;
            }

            div[data-testid="stFormSubmitButton"] {
                max-width: 42rem;
                margin: 0.1rem auto 0 auto;
            }

            div[data-testid="stFormSubmitButton"] > button {
                width: 100%;
                border-radius: 12px;
                min-height: 48px;
                border: 1px solid transparent;
                background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%);
                color: #ffffff;
                font-weight: 700;
                letter-spacing: 0.01em;
                box-shadow: 0 8px 20px rgba(29, 78, 216, 0.3);
                transition: transform 0.18s ease, box-shadow 0.18s ease, filter 0.18s ease;
            }

            div[data-testid="stFormSubmitButton"] > button:hover {
                transform: translateY(-1px) scale(1.01);
                filter: brightness(1.05);
                box-shadow: 0 10px 22px rgba(29, 78, 216, 0.35);
            }

            div.stButton > button:hover {
                transform: translateY(-1px) scale(1.01);
                filter: brightness(1.05);
                box-shadow: 0 10px 22px rgba(29, 78, 216, 0.35);
            }

            div[data-testid="stSpinner"] > div {
                border-radius: 10px;
                border: 1px solid rgba(29, 78, 216, 0.18);
                background: rgba(255, 255, 255, 0.78);
                padding: 0.45rem 0.65rem;
            }

            @keyframes fadeInUp {
                from {
                    opacity: 0;
                    transform: translateY(8px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }

            @media (prefers-color-scheme: dark) {
                :root {
                    --bg-main: radial-gradient(circle at top right, #0f172a 0%, #111827 45%, #0b1220 100%);
                    --card-bg: rgba(17, 24, 39, 0.78);
                    --card-border: rgba(148, 163, 184, 0.2);
                    --text-main: #e5e7eb;
                    --text-muted: #cbd5e1;
                    --accent: #60a5fa;
                    --accent-strong: #93c5fd;
                    --accent-soft: rgba(96, 165, 250, 0.16);
                    --shadow-soft: 0 14px 38px rgba(0, 0, 0, 0.45);
                }

                .result-card {
                    background: linear-gradient(135deg, rgba(30, 58, 138, 0.36) 0%, rgba(15, 23, 42, 0.56) 100%);
                    border: 1px solid rgba(96, 165, 250, 0.34);
                }

                .section-label,
                .result-title,
                .result-text,
                .hero-title {
                    color: #e2e8f0;
                }

                div[data-testid="stTextInput"] input {
                    background: rgba(2, 6, 23, 0.55);
                    color: #e5e7eb;
                    border: 1px solid rgba(148, 163, 184, 0.35);
                }

                div[data-testid="stSpinner"] > div {
                    background: rgba(15, 23, 42, 0.75);
                    border: 1px solid rgba(96, 165, 250, 0.28);
                }
            }

            @media (max-width: 768px) {
                .block-container {
                    padding-top: 0.85rem;
                }

                .hero-title {
                    font-size: 1.45rem;
                }

                .hero-card {
                    padding: 1rem;
                }

                .content-shell {
                    width: 100%;
                    padding: 0 0.2rem;
                }

                .section-label {
                    font-size: 0.92rem;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_resources() -> Tuple[
    Optional[object],
    Optional[object],
    Optional[int],
    Optional[Callable],
    Optional[Callable],
    Optional[int],
    List[str],
]:
    """Load and cache model artifacts for repeated predictions."""
    missing_files: List[str] = []

    for path in [MODEL_PATH, TOKENIZER_PATH, MAX_LEN_PATH]:
        if not path.exists():
            missing_files.append(path.name)

    if missing_files:
        return None, None, None, None, None, None, missing_files

    try:
        tensorflow_module = importlib.import_module("tensorflow")
        tensorflow_models = importlib.import_module("tensorflow.keras.models")
        tensorflow_sequence = importlib.import_module("tensorflow.keras.preprocessing.sequence")
        load_model = tensorflow_models.load_model
        keras_pad_sequences = tensorflow_sequence.pad_sequences
    except Exception as exc:
        return (
            None,
            None,
            None,
            None,
            None,
            None,
            [
                "TensorFlow import failed in the current environment.",
                f"Details: {exc}",
            ],
        )

    try:
        model = load_model(MODEL_PATH, compile=False)
        with TOKENIZER_PATH.open("rb") as tokenizer_file:
            tokenizer = pickle.load(tokenizer_file)
        with MAX_LEN_PATH.open("rb") as max_len_file:
            max_len = pickle.load(max_len_file)

        model_input_len = int(model.input_shape[1])

        @tensorflow_module.function(reduce_retracing=True)
        def predict_step(x):
            return model(x, training=False)

        # One warmup pass significantly reduces first prediction latency.
        warmup = tensorflow_module.zeros((1, model_input_len), dtype=tensorflow_module.int32)
        _ = predict_step(warmup)
    except Exception as exc:
        return None, None, None, None, None, None, [f"Resource loading failed: {exc}"]

    return model, tokenizer, int(max_len), keras_pad_sequences, predict_step, model_input_len, []


def find_next_word(predicted_index: int, tokenizer: object) -> str:
    """Return a word for the predicted token index."""
    index_word: Dict[int, str] = getattr(tokenizer, "index_word", {})
    return index_word.get(predicted_index, "")


def predict_words(
    seed_text: str,
    next_words: int,
    model: object,
    tokenizer: object,
    max_len: int,
    pad_fn: Callable,
    predict_fn: Optional[Callable] = None,
    model_input_len: Optional[int] = None,
) -> Tuple[str, List[str]]:
    """Generate one or more next words using the trained LSTM model."""
    generated_words: List[str] = []
    current_text = seed_text.strip()

    if model_input_len is None:
        try:
            model_input_len = int(model.input_shape[1])
        except Exception:
            model_input_len = max(1, int(max_len) - 1)

    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([current_text])[0]
        if not token_list:
            break

        padded = np.asarray(
            pad_fn([token_list], maxlen=model_input_len, padding="pre"),
            dtype=np.int32,
        )

        if predict_fn is not None:
            probabilities = predict_fn(padded).numpy()[0]
        else:
            probabilities = model(padded, training=False).numpy()[0]

        predicted_index = int(np.argmax(probabilities))
        next_word = find_next_word(predicted_index, tokenizer)

        if not next_word:
            break

        generated_words.append(next_word)
        current_text = f"{current_text} {next_word}".strip()

    return current_text, generated_words


def render_header() -> None:
    """Render the top hero section."""
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title-row">
                <span class="mat-icon" style="font-size: 30px; color: var(--accent);">neurology</span>
                <h1 class="hero-title">Next Word Prediction using LSTM</h1>
            </div>
            <div class="hero-divider"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result(seed_text: str, full_text: str, generated_words: List[str]) -> None:
    """Render prediction result card."""
    if not generated_words:
        st.warning("No word could be generated from this input. Try a longer sentence.")
        return

    if len(generated_words) == 1:
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-title">
                    <span class="mat-icon">trending_flat</span>
                    <span>Predicted Word:</span>
                </div>
                <p class="result-text"><span class="generated-word">{generated_words[0]}</span></p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    highlighted_sentence = seed_text
    for word in generated_words:
        highlighted_sentence += f" <span class='generated-word'>{word}</span>"

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-title">
                <span class="mat-icon">auto_awesome</span>
                <span>Generated Sentence:</span>
            </div>
            <p class="result-text">{highlighted_sentence}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    """Application entry point."""
    apply_styles()
    render_header()

    st.markdown('<div class="content-shell">', unsafe_allow_html=True)
    try:
        if "prediction_cache" not in st.session_state:
            st.session_state["prediction_cache"] = {}

        st.markdown(
            """
            <div class="section-label">
                <span class="mat-icon">edit_note</span>
                <span>Enter your sentence</span>
            </div>
            <div class="section-note">Provide a seed phrase to generate the next word.</div>
            """,
            unsafe_allow_html=True,
        )

        with st.form("prediction_form", clear_on_submit=False):
            seed_text = st.text_input(
                "Enter your sentence",
                value="",
                placeholder="Start typing your sentence...",
                label_visibility="collapsed",
            )

            st.markdown(
                """
                <div class="section-label">
                    <span class="mat-icon">tune</span>
                    <span>Number of words to generate</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            n_words = st.slider(
                "Number of words to generate",
                min_value=1,
                max_value=5,
                value=1,
                label_visibility="collapsed",
            )

            predict_clicked = st.form_submit_button(
                "Predict",
                icon=":material/arrow_forward:",
                use_container_width=True,
            )

        if predict_clicked:
            clean_text = seed_text.strip()
            if not clean_text:
                st.warning("Please enter a sentence before prediction.")
            else:
                cache_key = (clean_text, int(n_words))

                if cache_key in st.session_state["prediction_cache"]:
                    generated_sentence, generated_words = st.session_state["prediction_cache"][cache_key]
                else:
                    with st.spinner("Loading model and generating prediction..."):
                        model, tokenizer, max_len, pad_fn, predict_fn, model_input_len, load_errors = load_resources()

                        if load_errors:
                            st.error("Required model files are missing or invalid.")
                            st.info("Expected files: lstm_model.h5, tokenizer.pkl, max_len.pkl")
                            for err in load_errors:
                                st.write(f"- {err}")
                            return

                        generated_sentence, generated_words = predict_words(
                            seed_text=clean_text,
                            next_words=n_words,
                            model=model,
                            tokenizer=tokenizer,
                            max_len=max_len,
                            pad_fn=pad_fn,
                            predict_fn=predict_fn,
                            model_input_len=model_input_len,
                        )
                    st.session_state["prediction_cache"][cache_key] = (generated_sentence, generated_words)

                render_result(clean_text, generated_sentence, generated_words)
    finally:
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
