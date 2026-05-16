"""
pages/3_📂_Batch.py
Xử lý phản hồi hàng loạt từ file CSV/Excel/TXT.
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_loader import get_model_and_tokenizer
from utils.visualizer import sentiment_donut, aspect_bar_chart

st.set_page_config(page_title="Batch – TLU Analytics", page_icon="📂", layout="wide")

def load_css():
    css_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
load_css()

with st.sidebar:
    logo_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Logo-DH-Thuy-Loi.webp")
    if os.path.exists(logo_path):
        st.image(logo_path, width=90)
    st.markdown("<p style='color:#a5b4fc;font-weight:700;'>TLU Analytics</p>", unsafe_allow_html=True)

model, tokenizer = get_model_and_tokenizer()
if model:
    st.sidebar.success("🟢 AI Model Online")
else:
    st.sidebar.error("🔴 Model Offline")

ASPECT_COLS = ['Lecturer_Sentiment', 'Training_Sentiment', 'Facility_Sentiment', 'Others_Sentiment']

# ── Page Header ──────────────────────────────────────────────────────────────
st.markdown("""
<div class='page-header'>
    <div style='font-size:2.5rem;'>📂</div>
    <div>
        <h1 class='page-title'>Xử lý Hàng loạt</h1>
        <p class='page-subtitle'>Upload file và phân tích toàn bộ phản hồi sinh viên cùng lúc</p>
    </div>
</div>
""", unsafe_allow_html=True)

if not model:
    st.error("❌ Không thể tải mô hình AI.")
    st.stop()

def predict_batch(df, text_col):
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

# ── Upload ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class='chart-box'>
    <h3 style='color:#0f172a;margin-bottom:4px;'>📤 Tải lên File Phản hồi</h3>
    <p style='color:#64748b;font-size:0.85rem;'>Hỗ trợ: CSV, Excel (.xlsx), TXT (mỗi dòng một câu)</p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Chọn file", type=["csv", "xlsx", "txt"],
    help="Mỗi dòng là một câu phản hồi của sinh viên",
)

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df_raw = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            df_raw = pd.read_excel(uploaded_file)
        else:
            content = uploaded_file.read().decode("utf-8")
            df_raw = pd.DataFrame({"Feedback": [l.strip() for l in content.split("\n") if l.strip()]})
    except Exception as e:
        st.error(f"Lỗi đọc file: {e}")
        st.stop()

    st.success(f"✅ Đã tải: **{uploaded_file.name}** — {len(df_raw):,} dòng dữ liệu")
    st.dataframe(df_raw.head(5), use_container_width=True)

    text_col = st.selectbox("📌 Chọn cột chứa câu phản hồi:", df_raw.columns)

    col1, col2 = st.columns([1, 3])
    with col1:
        run_btn = st.button("🚀 Bắt đầu phân tích", type="primary", use_container_width=True)

    if run_btn:
        progress_bar = st.progress(0, text="Đang chuẩn bị...")
        status = st.empty()

        # Xử lý theo batch nhỏ để cập nhật progress bar
        batch_size = 50
        df_work = df_raw.copy()
        n = len(df_work)
        results_parts = []

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            chunk = df_work.iloc[start:end].copy()
            chunk_result = predict_batch(chunk, text_col)
            results_parts.append(chunk_result)
            progress = int((end / n) * 100)
            progress_bar.progress(progress, text=f"Đang xử lý {end}/{n} câu...")
            time.sleep(0.05)

        df_result = pd.concat(results_parts, ignore_index=True)
        if text_col != "Feedback":
            df_result = df_result.rename(columns={text_col: "Feedback"})

        progress_bar.progress(100, text="✅ Hoàn tất!")
        st.session_state['batch_result'] = df_result
        status.success(f"🎉 Đã phân tích xong {n:,} câu phản hồi!")

# ── Results ──────────────────────────────────────────────────────────────────
if 'batch_result' in st.session_state:
    df = st.session_state['batch_result']
    st.markdown("<div class='gradient-divider'></div>", unsafe_allow_html=True)
    st.markdown("### 📊 Kết quả Phân tích")

    # KPI nhanh
    all_s = df[ASPECT_COLS].melt().dropna()
    total = len(all_s)
    pos_pct = (all_s['value'] == "Tích cực 😄").sum() / total * 100 if total > 0 else 0
    neg_pct = (all_s['value'] == "Tiêu cực 😠").sum() / total * 100 if total > 0 else 0

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f"""<div class='kpi-card' style='border-left-color:#4f46e5;'>
            <div class='kpi-title'>Tổng câu</div><div class='kpi-value'>{len(df):,}</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class='kpi-card' style='border-left-color:#10b981;'>
            <div class='kpi-title'>Tích cực</div><div class='kpi-value'>{pos_pct:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class='kpi-card' style='border-left-color:#ef4444;'>
            <div class='kpi-title'>Tiêu cực</div><div class='kpi-value'>{neg_pct:.1f}%</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='chart-box'>", unsafe_allow_html=True)
        st.plotly_chart(sentiment_donut(all_s['value']), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='chart-box'>", unsafe_allow_html=True)
        st.plotly_chart(aspect_bar_chart(df, ASPECT_COLS), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Bảng có màu
    def color_sentiment(val):
        if pd.isna(val): return ''
        if "Tiêu cực" in str(val): return 'background-color:#fee2e2;color:#991b1b;font-weight:600'
        if "Tích cực" in str(val): return 'background-color:#dcfce7;color:#166534;font-weight:600'
        if "Trung lập" in str(val): return 'background-color:#fef9c3;color:#854d0e;font-weight:600'
        return ''

    st.markdown("<div class='chart-box'>", unsafe_allow_html=True)
    st.markdown("#### 📄 Bảng Kết quả Chi tiết")
    st.dataframe(
        df.style.map(color_sentiment, subset=ASPECT_COLS),
        use_container_width=True, height=420
    )

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Tải xuống CSV", data=csv, file_name="tlu_batch_results.csv", mime="text/csv")
    with col_dl2:
        try:
            import io
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Results')
            st.download_button("📥 Tải xuống Excel", data=buffer.getvalue(),
                               file_name="tlu_batch_results.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except:
            pass
    st.markdown("</div>", unsafe_allow_html=True)
