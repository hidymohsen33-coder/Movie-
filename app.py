"""
Movie Review Sentiment Intelligence — Streamlit demo
Fine-tuned BERT (bert-base-uncased) on the Stanford IMDB dataset,
plus a keyword-based aspect breakdown (acting / plot / music / visuals / directing).
"""

import re
import time

import streamlit as st
import torch
import plotly.graph_objects as go
from transformers import BertTokenizer, BertForSequenceClassification

# ------------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Sentiment Intelligence | BERT",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# Config — EDIT THIS after you push your model to the Hugging Face Hub
# ------------------------------------------------------------------
MODEL_REPO = "your-username/bert-imdb-sentiment"   # <-- change this
MAX_LENGTH = 256
LABEL_MAP = {0: "Negative", 1: "Positive"}

# Reported test-set metrics from the notebook — update with your real numbers
REPORTED_METRICS = {
    "Accuracy": 0.93,
    "Precision": 0.93,
    "Recall": 0.93,
    "F1-Score": 0.93,
}

ASPECTS = {
    "Acting":    ["acting", "actor", "actress", "performance", "cast", "role", "played"],
    "Plot":      ["plot", "story", "storyline", "script", "writing", "screenplay", "narrative"],
    "Music":     ["music", "score", "soundtrack", "song"],
    "Visuals":   ["visual", "cinematography", "effects", "cgi", "scenery", "shot"],
    "Directing": ["director", "directing", "directed"],
}

SAMPLE_REVIEWS = [
    "This movie was absolutely amazing. The acting was superb and the story kept me hooked until the last minute.",
    "Terrible film. I wasted two hours of my life waiting for something to happen.",
    "The acting was superb and the performances were unforgettable, but the plot was predictable and dragged on for too long.",
    "A visual masterpiece — the cinematography was breathtaking and the soundtrack was haunting, though the script felt weak.",
]

# ------------------------------------------------------------------
# Styling
# ------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        border-radius: 14px;
        padding: 18px 20px;
        border: 1px solid #2d3748;
    }
    .result-positive {
        background: linear-gradient(135deg, #064e3b 0%, #065f46 100%);
        border-left: 6px solid #10b981;
        border-radius: 10px;
        padding: 22px 26px;
    }
    .result-negative {
        background: linear-gradient(135deg, #4c0519 0%, #7f1d1d 100%);
        border-left: 6px solid #ef4444;
        border-radius: 10px;
        padding: 22px 26px;
    }
    .aspect-chip {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.85rem;
        margin: 3px 4px 3px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# Model loading (cached — loads once per session)
# ------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    tokenizer = BertTokenizer.from_pretrained(MODEL_REPO)
    model = BertForSequenceClassification.from_pretrained(MODEL_REPO)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tokenizer, model, device


def clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict_sentiment(text, tokenizer, model, device):
    text = clean_text(text)
    inputs = tokenizer(
        text, max_length=MAX_LENGTH, padding="max_length",
        truncation=True, return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        prediction = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][prediction].item()
    return {
        "label": LABEL_MAP[prediction],
        "confidence": confidence,
        "probabilities": {
            "Negative": probs[0][0].item(),
            "Positive": probs[0][1].item(),
        },
    }


def split_clauses(text):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    clauses = []
    for s in sentences:
        parts = re.split(
            r"\b(?:but|though|although|however|yet|while|whereas)\b|,",
            s, flags=re.IGNORECASE,
        )
        clauses.extend(p.strip() for p in parts if p.strip())
    return clauses


def aspect_sentiment(review, tokenizer, model, device):
    review = clean_text(review)
    clauses = split_clauses(review)
    results = {}
    for aspect, keywords in ASPECTS.items():
        matched = [
            c for c in clauses
            if any(re.search(r"\b" + re.escape(k) + r"\b", c.lower()) for k in keywords)
        ]
        if matched:
            pred = predict_sentiment(" ".join(matched), tokenizer, model, device)
            results[aspect] = pred
    return results


def gauge_chart(confidence, label):
    color = "#10b981" if label == "Positive" else "#ef4444"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        number={"suffix": "%", "font": {"size": 34, "color": "white"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "white"},
            "bar": {"color": color},
            "bgcolor": "#1f2937",
            "borderwidth": 0,
        },
    ))
    fig.update_layout(
        height=220, margin=dict(l=20, r=20, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", font={"color": "white"},
    )
    return fig


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🎬 Sentiment Intelligence")
    st.caption("Fine-tuned BERT · Stanford IMDB dataset · 50,000 reviews")
    st.divider()
    st.markdown("### Model")
    st.write("**Base model:** bert-base-uncased")
    st.write("**Task:** Binary sentiment classification")
    st.write("**Extra:** Aspect-based breakdown (acting, plot, music, visuals, directing)")
    st.divider()
    st.markdown("### Reported test-set performance")
    for k, v in REPORTED_METRICS.items():
        st.write(f"**{k}:** {v:.1%}")
    st.divider()
    st.caption("Built for demo purposes. Not a substitute for professional review-moderation tooling.")

# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
st.markdown("# 🎬 Movie Review Sentiment Intelligence")
st.markdown(
    "Analyze customer or audience reviews at a glance — overall sentiment plus a "
    "breakdown by **acting, plot, music, visuals, and directing**, powered by a "
    "fine-tuned BERT model."
)

col_top = st.columns(4)
metric_labels = list(REPORTED_METRICS.items())
for col, (name, val) in zip(col_top, metric_labels):
    with col:
        st.markdown(
            f"""<div class="metric-card">
                <div style="color:#9ca3af; font-size:0.85rem;">{name}</div>
                <div style="color:white; font-size:1.6rem; font-weight:700;">{val:.1%}</div>
                </div>""",
            unsafe_allow_html=True,
        )

st.write("")

# ------------------------------------------------------------------
# Try loading the model — show a friendly message if not configured yet
# ------------------------------------------------------------------
model_ready = True
try:
    with st.spinner("Loading model..."):
        tokenizer, model, device = load_model()
except Exception as e:
    model_ready = False
    st.error(
        "⚠️ Couldn't load the model yet. Set `MODEL_REPO` at the top of `app.py` to your "
        "Hugging Face model repository (e.g. after running `push_to_hub`), then redeploy.\n\n"
        f"Technical detail: {e}"
    )

# ------------------------------------------------------------------
# Main interaction
# ------------------------------------------------------------------
tab_single, tab_batch = st.tabs(["🔍 Analyze a review", "📦 Batch analysis"])

with tab_single:
    st.markdown("### Try a sample review")
    sample_cols = st.columns(len(SAMPLE_REVIEWS))
    chosen_sample = None
    for i, (col, sample) in enumerate(zip(sample_cols, SAMPLE_REVIEWS)):
        with col:
            if st.button(f"Example {i+1}", use_container_width=True, key=f"sample_{i}"):
                chosen_sample = sample

    default_text = chosen_sample if chosen_sample else ""
    review_text = st.text_area(
        "Paste a review or customer comment",
        value=default_text,
        height=140,
        placeholder="e.g. 'The acting was fantastic, but the plot felt rushed at the end.'",
    )

    analyze_clicked = st.button("Analyze sentiment", type="primary", disabled=not model_ready)

    if analyze_clicked and review_text.strip():
        with st.spinner("Analyzing..."):
            start = time.time()
            result = predict_sentiment(review_text, tokenizer, model, device)
            aspects = aspect_sentiment(review_text, tokenizer, model, device)
            elapsed = time.time() - start

        css_class = "result-positive" if result["label"] == "Positive" else "result-negative"
        emoji = "😊" if result["label"] == "Positive" else "😞"

        res_col, gauge_col = st.columns([2, 1])
        with res_col:
            st.markdown(
                f"""<div class="{css_class}">
                    <div style="font-size:1.4rem; font-weight:700; color:white;">
                        {emoji} Overall sentiment: {result['label']}
                    </div>
                    <div style="color:#e5e7eb; margin-top:6px;">
                        Confidence: {result['confidence']:.1%} · analyzed in {elapsed:.2f}s
                    </div>
                    </div>""",
                unsafe_allow_html=True,
            )
        with gauge_col:
            st.plotly_chart(gauge_chart(result["confidence"], result["label"]), use_container_width=True)

        if aspects:
            st.markdown("#### Aspect breakdown")
            chips = ""
            for aspect, info in aspects.items():
                color = "#065f46" if info["label"] == "Positive" else "#7f1d1d"
                chips += (
                    f'<span class="aspect-chip" style="background:{color}; color:white;">'
                    f'{aspect}: {info["label"]} ({info["confidence"]:.0%})</span>'
                )
            st.markdown(chips, unsafe_allow_html=True)
        else:
            st.caption("No recognized aspects (acting, plot, music, visuals, directing) were mentioned explicitly.")

    elif analyze_clicked:
        st.warning("Please enter a review first.")

with tab_batch:
    st.markdown("### Analyze many reviews at once")
    st.caption("Paste one review per line, or upload a .txt/.csv file with a single review per row.")
    batch_text = st.text_area("One review per line", height=180)
    uploaded = st.file_uploader("...or upload a file", type=["txt", "csv"])

    run_batch = st.button("Run batch analysis", disabled=not model_ready)

    if run_batch:
        lines = []
        if uploaded is not None:
            content = uploaded.read().decode("utf-8", errors="ignore")
            lines = [l.strip() for l in content.splitlines() if l.strip()]
        elif batch_text.strip():
            lines = [l.strip() for l in batch_text.splitlines() if l.strip()]

        if not lines:
            st.warning("Add at least one review.")
        else:
            with st.spinner(f"Analyzing {len(lines)} reviews..."):
                rows = []
                for line in lines:
                    r = predict_sentiment(line, tokenizer, model, device)
                    rows.append({
                        "Review": line[:120] + ("..." if len(line) > 120 else ""),
                        "Sentiment": r["label"],
                        "Confidence": f"{r['confidence']:.1%}",
                    })
            st.dataframe(rows, use_container_width=True)
            pos = sum(1 for r in rows if r["Sentiment"] == "Positive")
            neg = len(rows) - pos
            c1, c2 = st.columns(2)
            c1.metric("Positive reviews", pos)
            c2.metric("Negative reviews", neg)

st.divider()
st.caption(
    "Model: fine-tuned bert-base-uncased on the Stanford IMDB Large Movie Review Dataset "
    "(50,000 labeled reviews). Aspect detection is keyword-based and works best on English text."
)
