"""
pages/2_⚡_Predict.py
Trang kiểm thử từng câu với Hybrid ABSA pipeline.
"""
import streamlit as st
import os
import sys
import time
import base64

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_loader import get_model_and_tokenizer
from utils.absa_pipeline import run_absa_pipeline
from utils.visualizer import confidence_bar

st.set_page_config(page_title="Dự đoán từng câu – TLU Analytics", page_icon="⚡", layout="wide")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_css():
    css_path = os.path.join(ROOT_DIR, "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
load_css()

@st.cache_data
def get_base64_image(file_path):
    if not os.path.exists(file_path):
        return ""
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# Logo trên cùng sidebar
_logo_path = os.path.join(ROOT_DIR, "Logo-DH-Thuy-Loi.webp")

_logo_b64 = get_base64_image(_logo_path)
with st.sidebar:
    st.markdown(f"""
    <div style='display:flex;align-items:center;gap:12px;padding:4px 0 10px;'>
        <img src='data:image/webp;base64,{_logo_b64}'
             style='width:48px;height:48px;object-fit:contain;border-radius:50%;flex-shrink:0;'>
        <div>
            <p style='font-size:1rem;font-weight:800;color:#e0e7ff;margin:0;line-height:1.2;'>TLU Analytics</p>
            <p style='font-size:0.7rem;color:#a5b4fc;margin:0;'>Phân tích phản hồi sinh viên</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<hr style='border-color:rgba(255,255,255,0.15);margin:4px 0 12px;'>", unsafe_allow_html=True)


model, tokenizer = get_model_and_tokenizer()
if model:
    st.sidebar.success("🟢 Mô hình đang hoạt động")
else:
    st.sidebar.error("🔴 Mô hình ngoại tuyến")

# Câu mẫu gợi ý
EXAMPLE_SENTENCES = [
    "Cô dạy nhiệt tình nhưng phòng hơi nóng, chương trình đào tạo tốt",
    "Giảng viên tuyệt vời, tài liệu quá cũ",
    "Phòng học mát mẻ nhưng thầy khó tính",
    "Máy tính cùi bắp, học chán nhưng thầy dạy tốt",
    "Chương trình hay, trường đẹp, cô giáo tuyệt vời",
    "Wifi trường chập chờn, ảnh hưởng việc học",
    "Thầy giải thích dễ hiểu nhưng điều hòa bị hỏng",
]

# ── Page Header ──────────────────────────────────────────────────────────────
st.markdown("""
<div class='page-header'>
    <div style='font-size:2.5rem;'>⚡</div>
    <div>
        <h1 class='page-title'>Dự đoán từng câu</h1>
        <p class='page-subtitle'>Nhập câu tiếng Việt và nhận kết quả dự đoán đa khía cạnh ngay lập tức</p>
    </div>
</div>
""", unsafe_allow_html=True)

if not model:
    st.error("❌ Không thể tải mô hình. Kiểm tra file `sentiment_cnn_weights.weights.h5`.")
    st.stop()

# ── Input Form ───────────────────────────────────────────────────────────────
with st.form("predict_form"):
    text_input = st.text_area(
        "✍️ Nhập phản hồi sinh viên:",
        placeholder="Ví dụ: Cô dạy nhiệt tình nhưng phòng hơi nóng...",
        height=110,
    )
    col_btn, col_ex = st.columns([2, 5])
    with col_btn:
        submitted = st.form_submit_button("▶ Bắt đầu dự đoán", type="primary", use_container_width=True)

# Gợi ý câu mẫu
with st.expander("💡 Câu mẫu gợi ý", expanded=False):
    for i, ex in enumerate(EXAMPLE_SENTENCES):
        if st.button(f"▶ {ex}", key=f"ex_{i}"):
            text_input = ex
            submitted = True

# ── Prediction ───────────────────────────────────────────────────────────────
if submitted and text_input.strip():
    with st.spinner("🤖 Đang dự đoán..."):
        time.sleep(0.3)  # UX: nhỏ delay để spinner hiển thị
        results = run_absa_pipeline(text_input.strip(), model, tokenizer)

    st.markdown("<div class='gradient-divider'></div>", unsafe_allow_html=True)

    if not results:
        st.info("ℹ️ Mô hình không phát hiện khía cạnh cụ thể nào trong câu này.")
    else:
        st.markdown(f"""
        <div style='background:#f0fdf4;border:1px solid #bbf7d0;border-radius:12px;padding:16px 20px;margin-bottom:20px;'>
            <strong style='color:#166534;'>📝 Câu đã nhập:</strong>
            <span style='color:#1e293b;'> "{text_input.strip()}"</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"#### 🎯 Phát hiện được **{len(results)}** khía cạnh:")

        cols = st.columns(len(results))
        for i, result in enumerate(results):
            aspect = result["aspect"]
            sentiment = result["sentiment"]
            clause = result.get("clause", "")
            color = sentiment["color"]

            with cols[i]:
                st.markdown(f"""
                <div class='aspect-card' style='border-top-color:{color};'>
                    <div class='aspect-name'>{aspect}</div>
                    <div class='aspect-sentiment' style='color:{color};'>{sentiment["label"]}</div>
                </div>
                """, unsafe_allow_html=True)

                # fig = confidence_bar(sentiment["distribution"])
                # st.plotly_chart(fig, use_container_width=True)

    # ── Prediction History ────────────────────────────────────────────────────
    if 'predict_history' not in st.session_state:
        st.session_state['predict_history'] = []

    if submitted and text_input.strip() and results:
        history_entry = {
            "text": text_input.strip(),
            "aspects": [(r["aspect"], r["sentiment"]["label"]) for r in results],
        }
        # Tránh duplicate
        if not st.session_state['predict_history'] or \
           st.session_state['predict_history'][0]['text'] != text_input.strip():
            st.session_state['predict_history'].insert(0, history_entry)
            st.session_state['predict_history'] = st.session_state['predict_history'][:20]

elif submitted:
    st.warning("⚠️ Vui lòng nhập câu phản hồi trước khi dự đoán.")

# ── History ──────────────────────────────────────────────────────────────────
if st.session_state.get('predict_history'):
    st.markdown("<div class='gradient-divider'></div>", unsafe_allow_html=True)
    st.markdown("#### 🕐 Lịch sử dự đoán")
    for entry in st.session_state['predict_history']:
        badge_html = ""
        for aspect, label in entry["aspects"]:
            if "Tích cực" in label:
                badge_html += f"<span class='badge-pos'>{aspect}: {label}</span> "
            elif "Tiêu cực" in label:
                badge_html += f"<span class='badge-neg'>{aspect}: {label}</span> "
            else:
                badge_html += f"<span class='badge-neu'>{aspect}: {label}</span> "
        st.markdown(f"""
        <div class='history-row'>
            <div style='color:#1e293b;margin-bottom:8px;font-size:0.9rem;'>"{entry['text']}"</div>
            <div>{badge_html}</div>
        </div>
        """, unsafe_allow_html=True)

    if st.button("🗑️ Xóa lịch sử", key="clear_history"):
        st.session_state['predict_history'] = []
        st.rerun()
