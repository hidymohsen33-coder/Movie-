import time
import re
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from transformers import pipeline

# ==========================================
# 1. Streamlit Page Configuration
# ==========================================
st.set_page_config(
    page_title="CineSense - ABSA & Sentiment Analysis",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Enterprise Grade UI)
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .header-title { text-align: center; font-size: 36px; font-weight: 800; color: #4F46E5; margin-bottom: 0px; }
    .header-sub { text-align: center; font-size: 16px; color: #6B7280; margin-bottom: 25px; }
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .pos-card { border-top: 5px solid #10B981; }
    .neg-card { border-top: 5px solid #EF4444; }
    .card-label { font-size: 13px; font-weight: 600; color: #6B7280; text-transform: uppercase; }
    .card-val { font-size: 28px; font-weight: 800; margin: 5px 0; }
    .pos-text { color: #10B981; }
    .neg-text { color: #EF4444; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Cached Pipeline Loader
# ==========================================
@st.cache_resource
def load_sentiment_pipeline():
    model_name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    return pipeline("text-classification", model=model_name, top_k=None)

classifier = load_sentiment_pipeline()

# Helper Inference Function
def predict_text(text):
    results = classifier(text)
    if isinstance(results, list) and len(results) > 0:
        raw = results[0] if isinstance(results[0], list) else results
        scores = {res['label']: res['score'] for res in raw}
        pos = scores.get('POSITIVE', scores.get('LABEL_1', 0.0))
        neg = scores.get('NEGATIVE', scores.get('LABEL_0', 0.0))
        label = "POSITIVE" if pos > neg else "NEGATIVE"
        return label, pos, neg
    return "UNKNOWN", 0.0, 0.0

# ==========================================
# 3. Aspect-Based Sentiment Analysis (ABSA) Engine
# ==========================================
ASPECT_KEYWORDS = {
    "acting": ["acting", "actor", "actress", "cast", "performance", "performances"],
    "plot": ["plot", "story", "script", "ending", "storyline", "writing"],
    "directing": ["directing", "director", "direction", "filmmaking"],
    "visuals": ["visuals", "cinematography", "effects", "vfx", "scenes", "camera"],
    "music": ["music", "soundtrack", "score", "songs", "sound"]
}

def analyze_aspects(text):
    # Split text into sub-sentences using conjunctions and punctuation
    sentences = re.split(r'[\.\!\?\,]|\bbut\b|\band\b', text, flags=re.IGNORECASE)
    aspect_results = {}

    for aspect, keywords in ASPECT_KEYWORDS.items():
        relevant_sentences = []
        for sent in sentences:
            if any(re.search(r'\b' + kw + r'\b', sent, re.IGNORECASE) for kw in keywords):
                relevant_sentences.append(sent.strip())

        if relevant_sentences:
            combined_sent = " ".join(relevant_sentences)
            label, pos_s, neg_s = predict_text(combined_sent)
            aspect_results[aspect] = {
                "sentiment": label,
                "positive_pct": round(pos_s * 100, 1),
                "negative_pct": round(neg_s * 100, 1),
                "mentioned": True
            }
        else:
            aspect_results[aspect] = {
                "sentiment": "Not Mentioned",
                "positive_pct": 0.0,
                "negative_pct": 0.0,
                "mentioned": False
            }

    return aspect_results

# ==========================================
# 4. Main Interface
# ==========================================
st.markdown("<h1 class='header-title'>🎬 CineSense</h1>", unsafe_allow_html=True)
st.markdown("<p class='header-sub'>Aspect-Based Sentiment Analysis (ABSA) Engine</p>", unsafe_allow_html=True)

st.sidebar.title("⚙️ Engine Specs")
st.sidebar.markdown("---")
st.sidebar.markdown("**Architecture:** Fine-Tuned BERT + ABSA")
st.sidebar.markdown("**Aspects Analyzed:** Acting, Plot, Directing, Visuals, Music")
st.sidebar.markdown("**Status:** 🟢 Live 24/7 API")

tab_single, tab_batch = st.tabs(["📄 Single Review & Aspect Breakdown", "📊 Batch Analytics"])

# ------------------------------------------
# TAB 1: Single Review Analysis with ABSA
# ------------------------------------------
with tab_single:
    st.subheader("Single Text & Aspect Breakdown")
    
    preset = st.selectbox(
        "Choose an example review or type custom input:",
        [
            "Custom Input",
            "Sample 1: The acting was superb but the story was dull and the ending was disappointing.",
            "Sample 2: The visuals were stunning and the music was great, but the plot made no sense.",
            "Sample 3: An absolute masterpiece with flawless direction and brilliant acting from start to finish."
        ]
    )

    if "Sample 1" in preset:
        default_val = "The acting was superb but the story was dull and the ending was disappointing."
    elif "Sample 2" in preset:
        default_val = "The visuals were stunning and the music was great, but the plot made no sense."
    elif "Sample 3" in preset:
        default_val = "An absolute masterpiece with flawless direction and brilliant acting from start to finish."
    else:
        default_val = "The acting was superb but the plot was predictable and boring."

    user_text = st.text_area("Enter a movie review:", value=default_val, height=90)

    if st.button("🚀 Analyze Sentiment & Aspects", type="primary"):
        if not user_text.strip():
            st.warning("Please enter text first.")
        else:
            # Overall Prediction
            overall_lbl, overall_pos, overall_neg = predict_text(user_text)
            aspect_data = analyze_aspects(user_text)

            st.markdown("---")
            
            # Overall Score Cards
            col_pos, col_neg, col_chart = st.columns([1, 1, 1.5])
            with col_pos:
                st.markdown(f"""
                <div class='metric-card pos-card'>
                    <div class='card-label'>Overall Positive</div>
                    <div class='card-val pos-text'>{overall_pos*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

            with col_neg:
                st.markdown(f"""
                <div class='metric-card neg-card'>
                    <div class='card-label'>Overall Negative</div>
                    <div class='card-val neg-text'>{overall_neg*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

            with col_chart:
                fig_overall = go.Figure(data=[go.Pie(
                    labels=['Positive', 'Negative'],
                    values=[overall_pos, overall_neg],
                    hole=.6,
                    marker_colors=['#10B981', '#EF4444'],
                    textinfo='percent+label'
                )])
                fig_overall.update_layout(showlegend=False, height=180, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig_overall, use_container_width=True)

            # Aspect-Based Breakdown Section
            st.markdown("### 🎯 Aspect Breakdown (Positive vs Negative %)")

            aspect_list = []
            for asp, res in aspect_data.items():
                if res['mentioned']:
                    aspect_list.append({
                        "Aspect": asp.capitalize(),
                        "Positive (%)": res['positive_pct'],
                        "Negative (%)": res['negative_pct'],
                        "Sentiment": res['sentiment']
                    })

            if aspect_list:
                df_aspects = pd.DataFrame(aspect_list)

                # Grouped Bar Chart for Aspect Percentages
                fig_bar = go.Figure()
                fig_bar.add_trace(go.Bar(
                    x=df_aspects['Aspect'],
                    y=df_aspects['Positive (%)'],
                    name='Positive %',
                    marker_color='#10B981',
                    text=df_aspects['Positive (%)'].apply(lambda x: f"{x}%"),
                    textposition='auto'
                ))
                fig_bar.add_trace(go.Bar(
                    x=df_aspects['Aspect'],
                    y=df_aspects['Negative (%)'],
                    name='Negative %',
                    marker_color='#EF4444',
                    text=df_aspects['Negative (%)'].apply(lambda x: f"{x}%"),
                    textposition='auto'
                ))

                fig_bar.update_layout(
                    barmode='group',
                    height=320,
                    margin=dict(l=20, r=20, t=20, b=20),
                    yaxis=dict(title="Percentage (%)", range=[0, 100]),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )

                col_a1, col_a2 = st.columns([1.5, 1])
                with col_a1:
                    st.plotly_chart(fig_bar, use_container_width=True)
                with col_a2:
                    st.dataframe(df_aspects, use_container_width=True, height=250)
            else:
                st.info("No specific movie aspects (acting, plot, directing, visuals, music) detected in this review.")

# ------------------------------------------
# TAB 2: Batch Analysis (Mention Frequency)
# ------------------------------------------
with tab_batch:
    st.subheader("Batch Review & Aspect Mention Frequency")
    
    default_batch = (
        "The acting was superb but the story was dull.\n"
        "Terrible film, bad plot, and horrible acting.\n"
        "The visuals were stunning and the music was incredible.\n"
        "Boring plot and predictable story from start to finish.\n"
        "Great directing and brilliant acting performances.\n"
        "Awful script, poor directing, but nice soundtrack."
    )
    batch_text = st.text_area("Enter multiple reviews (one per line):", value=default_batch, height=140)
    reviews_list = [line.strip() for line in batch_text.split("\n") if line.strip()]

    if st.button("📊 Analyze Batch Aspects", type="primary"):
        if not reviews_list:
            st.warning("Please enter text lines.")
        else:
            total_reviews = len(reviews_list)
            aspect_counts = {asp: 0 for asp in ASPECT_KEYWORDS.keys()}

            for rev in reviews_list:
                asp_res = analyze_aspects(rev)
                for asp, details in asp_res.items():
                    if details['mentioned']:
                        aspect_counts[asp] += 1

            # Build Frequency DataFrame
            freq_data = []
            for asp, count in aspect_counts.items():
                freq_data.append({
                    "Aspect": asp.capitalize(),
                    "Share of Reviews (%)": round((count / total_reviews) * 100, 1)
                })

            df_freq = pd.DataFrame(freq_data).sort_values(by="Share of Reviews (%)", ascending=False)

            st.markdown("---")
            st.markdown("### 📊 How Often Each Aspect Is Mentioned")

            fig_freq = px.bar(
                df_freq,
                x='Aspect',
                y='Share of Reviews (%)',
                text='Share of Reviews (%)',
                color_discrete_sequence=['#3B82F6']
            )
            fig_freq.update_traces(texttemplate='%{text}%', textposition='outside')
            fig_freq.update_layout(
                height=350,
                yaxis=dict(range=[0, 100]),
                margin=dict(l=20, r=20, t=30, b=20)
            )
            st.plotly_chart(fig_freq, use_container_width=True)
