"""
pages/1_📊_Dashboard.py
Dashboard phân tích tổng quan phản hồi sinh viên.
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_loader import get_model_and_tokenizer
from utils.absa_pipeline import run_absa_pipeline, SENTIMENT_MAP
from utils.visualizer import sentiment_donut, aspect_bar_chart
from utils.preprocessor import clean_text
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(page_title="Dashboard – TLU Analytics", page_icon="📊", layout="wide")

# Load CSS
def load_css():
    css_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
load_css()

# Sidebar branding
with st.sidebar:
    logo_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Logo-DH-Thuy-Loi.webp")
    if os.path.exists(logo_path):
        st.image(logo_path, width=90)
    st.markdown("<p style='color:#a5b4fc;font-weight:700;'>TLU Analytics</p>", unsafe_allow_html=True)
    model_status = st.empty()

model, tokenizer = get_model_and_tokenizer()
with model_status:
    if model:
        st.sidebar.success("🟢 AI Model Online")
    else:
        st.sidebar.error("🔴 Model Offline")

# ── Page Header ─────────────────────────────────────────────────────────────
st.markdown("""
<div class='page-header'>
    <div style='font-size:2.5rem;'>📊</div>
    <div>
        <h1 class='page-title'>Dashboard Phân tích</h1>
        <p class='page-subtitle'>Tổng quan cảm xúc và khía cạnh từ phản hồi sinh viên</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Aspect cols mapping ──────────────────────────────────────────────────────
ASPECT_COLS = ['Lecturer_Sentiment', 'Training_Sentiment', 'Facility_Sentiment', 'Others_Sentiment']
ASPECT_DISPLAY = {
    'Lecturer_Sentiment': 'Giảng viên',
    'Training_Sentiment': 'Chương trình',
    'Facility_Sentiment': 'Cơ sở vật chất',
    'Others_Sentiment': 'Khác',
}

def get_representative_sentences(sentences, top_n=5):
    sentences = [str(s) for s in sentences if str(s).strip()]
    if len(sentences) <= top_n:
        return sentences
    try:
        vectorizer = TfidfVectorizer(max_df=0.85)
        X = vectorizer.fit_transform(sentences)
        scores = np.array(X.sum(axis=1)).flatten()
        top_indices = scores.argsort()[-top_n:][::-1]
        return [sentences[i] for i in top_indices]
    except:
        return sentences[:top_n]

def predict_batch(df, text_col):
    """Batch predict dùng model branches."""
    if not model or not tokenizer:
        return df
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    texts = df[text_col].astype(str).tolist()
    seqs = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(seqs, maxlen=100, padding='post', truncating='post')
    preds = model.predict(padded, batch_size=64, verbose=0)
    sentiment_map_local = {0: "Tiêu cực 😠", 1: "Trung lập 😐", 2: "Tích cực 😄"}
    aspect_keys = ['Lecturer', 'Training', 'Facility', 'Others']
    for i, aspect in enumerate(aspect_keys):
        pred_classes = np.argmax(preds[i], axis=1)
        labels = [sentiment_map_local.get(cls) if cls < 3 else None for cls in pred_classes]
        df[f'{aspect}_Sentiment'] = labels
    return df

@st.cache_data
def load_demo_data():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    try:
        with open(os.path.join(data_dir, "test", "sents.txt"), "r", encoding="utf-8") as f:
            lines = f.readlines()
        df = pd.DataFrame({"Feedback": [line.strip() for line in lines[:500] if line.strip()]})
        return predict_batch(df, "Feedback")
    except Exception as e:
        return None

# ── Data Loading ─────────────────────────────────────────────────────────────
with st.expander("📂 Tải lên Dữ liệu mới", expanded=False):
    uploaded_file = st.file_uploader("Kéo thả CSV, XLSX hoặc TXT", type=["csv", "xlsx", "txt"])
    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df_upload = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            df_upload = pd.read_excel(uploaded_file)
        else:
            content = uploaded_file.read().decode("utf-8")
            df_upload = pd.DataFrame({"Feedback": [l.strip() for l in content.split("\n") if l.strip()]})
        text_col = st.selectbox("Chọn cột chứa câu phản hồi:", df_upload.columns)
        if st.button("🚀 Phân tích AI", type="primary"):
            with st.spinner("Đang chạy ABSA pipeline..."):
                results_df = predict_batch(df_upload, text_col)
                if text_col != "Feedback":
                    results_df = results_df.rename(columns={text_col: "Feedback"})
                st.session_state['dashboard_data'] = results_df
                st.success("✅ Phân tích hoàn tất!")

st.markdown("<div class='gradient-divider'></div>", unsafe_allow_html=True)

# Load dữ liệu
if 'dashboard_data' not in st.session_state:
    with st.spinner("Đang tải dữ liệu mẫu..."):
        st.session_state['dashboard_data'] = load_demo_data()

df = st.session_state.get('dashboard_data')

if df is not None and not df.empty:
    # ── KPI Cards ───────────────────────────────────────────────────────────
    all_sentiments = df[ASPECT_COLS].melt().dropna()
    pos_count = (all_sentiments['value'] == "Tích cực 😄").sum()
    neg_count = (all_sentiments['value'] == "Tiêu cực 😠").sum()
    total_mentions = len(all_sentiments)
    pos_pct = pos_count / total_mentions * 100 if total_mentions > 0 else 0
    neg_pct = neg_count / total_mentions * 100 if total_mentions > 0 else 0
    mention_counts = {col: df[col].notna().sum() for col in ASPECT_COLS}
    top_aspect_col = max(mention_counts, key=mention_counts.get)
    top_aspect = ASPECT_DISPLAY.get(top_aspect_col, top_aspect_col)

    k1, k2, k3, k4 = st.columns(4)
    kpi_data = [
        (k1, "#4f46e5", "📝", "Tổng phản hồi", f"{len(df):,}", "câu đã xử lý"),
        (k2, "#10b981", "😄", "Tích cực", f"{pos_pct:.1f}%", "trên tất cả khía cạnh"),
        (k3, "#ef4444", "😠", "Tiêu cực", f"{neg_pct:.1f}%", "cần cải thiện"),
        (k4, "#3b82f6", "🔥", "Top Khía cạnh", top_aspect, "được nhắc đến nhiều nhất"),
    ]
    for col, color, icon, title, value, sub in kpi_data:
        with col:
            st.markdown(f"""
            <div class='kpi-card' style='border-left-color:{color};'>
                <div class='kpi-icon'>{icon}</div>
                <div class='kpi-title'>{title}</div>
                <div class='kpi-value'>{value}</div>
                <div class='kpi-sub'>{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["📈 Báo cáo Tổng quan", "🗂️ Phân tích Chi tiết"])

    with tab1:
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown("<div class='chart-box'>", unsafe_allow_html=True)
            fig_pie = sentiment_donut(all_sentiments['value'], "Tổng quan Cảm xúc")
            st.plotly_chart(fig_pie, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown("<div class='chart-box'>", unsafe_allow_html=True)
            fig_bar = aspect_bar_chart(df, ASPECT_COLS)
            st.plotly_chart(fig_bar, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Word cloud
        try:
            from wordcloud import WordCloud
            import matplotlib.pyplot as plt
            st.markdown("<div class='chart-box'>", unsafe_allow_html=True)
            st.markdown("#### ☁️ Word Cloud phản hồi")
            all_text = ' '.join(df['Feedback'].astype(str).tolist())
            wc = WordCloud(width=900, height=300, background_color='white',
                           colormap='RdYlGn', max_words=100,
                           font_path=None).generate(all_text)
            fig_wc, ax = plt.subplots(figsize=(12, 4))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            fig_wc.patch.set_alpha(0)
            st.pyplot(fig_wc)
            st.markdown("</div>", unsafe_allow_html=True)
        except ImportError:
            st.info("💡 Cài thêm `wordcloud` để xem Word Cloud: `pip install wordcloud`")

    with tab2:
        st.markdown("<div class='chart-box'>", unsafe_allow_html=True)
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            selected_aspect = st.selectbox("Lọc theo Khía cạnh", ["Tất cả"] + list(ASPECT_DISPLAY.values()))
        with col_f2:
            selected_sentiment = st.selectbox("Lọc theo Cảm xúc", ["Tất cả", "Tích cực 😄", "Tiêu cực 😠", "Trung lập 😐"])

        filtered_df = df.copy()
        asp_col_map = {v: k for k, v in ASPECT_DISPLAY.items()}
        if selected_aspect != "Tất cả":
            col_name = asp_col_map.get(selected_aspect, "")
            if col_name:
                if selected_sentiment != "Tất cả":
                    filtered_df = filtered_df[filtered_df[col_name] == selected_sentiment]
                else:
                    filtered_df = filtered_df[filtered_df[col_name].notna()]
        elif selected_sentiment != "Tất cả":
            mask = pd.Series([False] * len(filtered_df), index=filtered_df.index)
            for c in ASPECT_COLS:
                mask = mask | (filtered_df[c] == selected_sentiment)
            filtered_df = filtered_df[mask]

        # Ý kiến tiêu biểu
        st.markdown("---")
        st.markdown("##### 💡 Ý kiến Tiêu biểu (AI trích xuất)")
        top_sents = get_representative_sentences(filtered_df['Feedback'].tolist(), top_n=5)
        for s in top_sents:
            st.markdown(f"> *\"{s}\"*")

        st.markdown("---")
        st.markdown("##### 📄 Bảng Dữ liệu Chi tiết")

        def color_sentiment(val):
            if pd.isna(val): return ''
            if "Tiêu cực" in str(val): return 'background-color:#fee2e2;color:#991b1b;font-weight:600'
            if "Tích cực" in str(val): return 'background-color:#dcfce7;color:#166534;font-weight:600'
            if "Trung lập" in str(val): return 'background-color:#fef9c3;color:#854d0e;font-weight:600'
            return ''

        st.dataframe(
            filtered_df.style.map(color_sentiment, subset=ASPECT_COLS),
            use_container_width=True, height=400
        )
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Xuất CSV", data=csv, file_name="tlu_absa_results.csv", mime="text/csv")
        st.markdown("</div>", unsafe_allow_html=True)
else:
    st.warning("⚠️ Chưa có dữ liệu. Vui lòng tải file lên hoặc kiểm tra thư mục `data/`.")
