import time
import streamlit as st
from transformers import pipeline

# ==========================================
# 1. Streamlit Page Configuration
# ==========================================
st.set_page_config(
    page_title="BERT Sentiment Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern Corporate Styling
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    div[data-testid="stMetricValue"] { font-size: 24px; font-weight: bold; color: #1E88E5; }
    div[data-testid="stMetricLabel"] { font-size: 14px; color: #555555; }
    .positive-box {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .negative-box {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .sub-header {
        color: #2c3e50;
        border-bottom: 2px solid #e0e0e0;
        padding-bottom: 5px;
        margin-top: 20px;
        margin-bottom: 15px;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Pipeline Loader (Loads your Custom Model)
# ==========================================
@st.cache_resource
def load_sentiment_pipeline():
    # ⚠️ استبدلي هذا السطر باسم المستخدم واسم النموذج الخاص بك على Hugging Face
    model_name = "your-username/bert-imdb-sentiment" 
    return pipeline("text-classification", model=model_name, return_all_scores=True)

classifier = load_sentiment_pipeline()

# ==========================================
# 3. Sidebar Configuration
# ==========================================
st.sidebar.title("⚙️ Model Configuration")
st.sidebar.markdown("---")
st.sidebar.subheader("Hyperparameters")
st.sidebar.text("Architecture: BERT Fine-Tuned")
st.sidebar.text("Dataset: Stanford IMDB")
st.sidebar.text("Status: Live 24/7 (Cloud Hosted)")

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Settings")
show_latency = st.sidebar.checkbox("Show Inference Latency", value=True)

# ==========================================
# 4. Header & Top Metrics
# ==========================================
st.title("🎬 Custom BERT Sentiment Dashboard")
st.markdown("Fine-tuned BERT Model deployed on Streamlit Cloud.")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="Model Architecture", value="Custom BERT")
with col2:
    st.metric(label="Dataset Size", value="50,000 Reviews")
with col3:
    st.metric(label="Status", value="🟢 Live 24/7")
with col4:
    st.metric(label="Hosting", value="Streamlit Cloud")

st.markdown("<h3 class='sub-header'>📝 Review Input & Inference</h3>", unsafe_allow_html=True)

preset_option = st.selectbox(
    "Choose a sample review or enter your own:",
    [
        "Custom Input",
        "Sample 1 (Positive): This movie was an absolute masterpiece! The acting was superb.",
        "Sample 2 (Negative): Absolutely terrible film. The plot made no sense and was very boring."
    ]
)

if "Sample 1" in preset_option:
    default_text = "This movie was an absolute masterpiece! The acting was superb."
elif "Sample 2" in preset_option:
    default_text = "Absolutely terrible film. The plot made no sense and was very boring."
else:
    default_text = "I really enjoyed this movie, the cinematography and direction were top notch."

user_review = st.text_area("Movie Review Text:", value=default_text, height=120)

# ==========================================
# 5. Prediction Logic
# ==========================================
if st.button("🚀 Analyze Sentiment", type="primary"):
    if user_review.strip() == "":
        st.warning("Please enter text to analyze.")
    else:
        start_time = time.time()
        results = classifier(user_review)[0]
        latency = (time.time() - start_time) * 1000

        scores = {res['label']: res['score'] for res in results}
        
        # دعم أسماء المخرجات المختلفة (LABEL_1/LABEL_0 أو POSITIVE/NEGATIVE)
        pos_score = scores.get('POSITIVE', scores.get('LABEL_1', 0.0))
        neg_score = scores.get('NEGATIVE', scores.get('LABEL_0', 0.0))
        
        is_positive = pos_score > neg_score

        col_res, col_prob = st.columns([1, 1])

        with col_res:
            st.markdown("<h4 class='sub-header'>Predicted Sentiment</h4>", unsafe_allow_html=True)
            if is_positive:
                st.markdown(
                    f"""
                    <div class='positive-box'>
                        <h2 style='color: #2e7d32; margin:0;'>POSITIVE 😃</h2>
                        <p style='margin:5px 0 0 0;'>Confidence: <b>{pos_score*100:.2f}%</b></p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div class='negative-box'>
                        <h2 style='color: #c62828; margin:0;'>NEGATIVE 😞</h2>
                        <p style='margin:5px 0 0 0;'>Confidence: <b>{neg_score*100:.2f}%</b></p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )

        with col_prob:
            st.markdown("<h4 class='sub-header'>Probability Distribution</h4>", unsafe_allow_html=True)
            st.write(f"**Negative:** {neg_score*100:.2f}%")
            st.progress(float(neg_score))
            st.write(f"**Positive:** {pos_score*100:.2f}%")
            st.progress(float(pos_score))

        if show_latency:
            st.markdown("<h4 class='sub-header'>🛠️ Technical Details</h4>", unsafe_allow_html=True)
            st.info(f"⚡ **Inference Latency:** {latency:.2f} ms")
