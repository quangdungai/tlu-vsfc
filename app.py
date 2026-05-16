"""
app.py — Entry Point cho TLU Feedback Analytics
Multipage Streamlit app với sidebar navigation.
"""
import streamlit as st
import os

# ── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TLU Feedback Analytics",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load Global CSS ─────────────────────────────────────────────────────────
def load_css():
    css_path = os.path.join("assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ── Sidebar Branding ────────────────────────────────────────────────────────
with st.sidebar:
    logo_path = "Logo-DH-Thuy-Loi.webp"
    if os.path.exists(logo_path):
        st.image(logo_path, width=90)

    st.markdown("""
    <div style='margin-top:-10px;'>
        <p style='font-size:1.1rem;font-weight:800;color:#e0e7ff;margin:0;'>TLU Analytics</p>
        <p style='font-size:0.75rem;color:#a5b4fc;margin:0;'>Phân tích phản hồi sinh viên</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:rgba(255,255,255,0.15);margin:16px 0;'>", unsafe_allow_html=True)
    st.markdown("<p style='color:#a5b4fc;font-size:0.75rem;font-weight:600;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;'>Navigation</p>", unsafe_allow_html=True)

# ── Home Page Content ───────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;padding:60px 20px 40px;'>
    <div style='font-size:4rem;margin-bottom:16px;'>🌊</div>
    <h1 style='font-size:2.5rem;font-weight:800;color:#0f172a;margin-bottom:8px;'>
        TLU Feedback Analytics
    </h1>
    <p style='font-size:1.1rem;color:#64748b;max-width:600px;margin:0 auto 32px;'>
        Hệ thống phân tích phản hồi sinh viên Trường Đại học Thủy Lợi<br>
        sử dụng AI – Aspect-Based Sentiment Analysis
    </p>
</div>
""", unsafe_allow_html=True)

# ── Feature Cards ───────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class='kpi-card' style='border-left-color:#4f46e5;text-align:center;'>
        <div style='font-size:2.5rem;margin-bottom:12px;'>📊</div>
        <div style='font-weight:700;color:#0f172a;font-size:1.1rem;margin-bottom:8px;'>Dashboard Phân tích</div>
        <div style='color:#64748b;font-size:0.85rem;'>Xem tổng quan sentiment, biểu đồ theo khía cạnh và các chỉ số KPI.</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class='kpi-card' style='border-left-color:#10b981;text-align:center;'>
        <div style='font-size:2.5rem;margin-bottom:12px;'>⚡</div>
        <div style='font-weight:700;color:#0f172a;font-size:1.1rem;margin-bottom:8px;'>Kiểm thử Từng câu</div>
        <div style='color:#64748b;font-size:0.85rem;'>Nhập câu tiếng Việt và nhận kết quả phân tích đa khía cạnh realtime.</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class='kpi-card' style='border-left-color:#f59e0b;text-align:center;'>
        <div style='font-size:2.5rem;margin-bottom:12px;'>📂</div>
        <div style='font-weight:700;color:#0f172a;font-size:1.1rem;margin-bottom:8px;'>Xử lý Hàng loạt</div>
        <div style='color:#64748b;font-size:0.85rem;'>Upload file CSV/Excel và phân tích toàn bộ phản hồi cùng lúc.</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div class='gradient-divider'></div>", unsafe_allow_html=True)

# ── How it Works ────────────────────────────────────────────────────────────
st.markdown("""
<h2 style='text-align:center;color:#0f172a;font-weight:700;margin-bottom:24px;'>
    Cách hệ thống hoạt động
</h2>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
steps = [
    ("1️⃣", "Nhập phản hồi", "Sinh viên nhập câu tiếng Việt bất kỳ"),
    ("2️⃣", "Tách mệnh đề", "AI tách theo từ nối: nhưng, tuy nhiên, còn..."),
    ("3️⃣", "Phát hiện khía cạnh", "Keyword matching → Giảng viên, Cơ sở, Chương trình..."),
    ("4️⃣", "Phân tích cảm xúc", "CNN + BiLSTM model predict Positive/Neutral/Negative"),
]
for col, (icon, title, desc) in zip([c1, c2, c3, c4], steps):
    with col:
        st.markdown(f"""
        <div style='text-align:center;padding:16px;'>
            <div style='font-size:2rem;margin-bottom:8px;'>{icon}</div>
            <div style='font-weight:700;color:#0f172a;margin-bottom:6px;font-size:0.95rem;'>{title}</div>
            <div style='color:#64748b;font-size:0.8rem;'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center;'>
    <p style='color:#94a3b8;font-size:0.8rem;'>
        Sử dụng menu bên trái để điều hướng → 
        <strong style='color:#4f46e5;'>📊 Dashboard</strong> · 
        <strong style='color:#10b981;'>⚡ Predict</strong> · 
        <strong style='color:#f59e0b;'>📂 Batch</strong>
    </p>
</div>
""", unsafe_allow_html=True)
