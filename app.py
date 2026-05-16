"""
app.py — Trang chủ TLU Feedback Analytics
"""
import streamlit as st
import os

st.set_page_config(
    page_title="TLU Feedback Analytics",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

def load_css():
    css_path = os.path.join("assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ── Sidebar ───────────────────────────────────────────────────────────────────
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
    st.markdown(
        "<p style='color:#a5b4fc;font-size:0.75rem;font-weight:600;"
        "text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;'>Điều hướng</p>",
        unsafe_allow_html=True
    )

# ── Trang chủ ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;padding:60px 20px 40px;'>
    <div style='font-size:4rem;margin-bottom:16px;'>🌊</div>
    <h1 style='font-size:2.5rem;font-weight:800;color:#0f172a;margin-bottom:8px;'>
        TLU Feedback Analytics
    </h1>
    <p style='font-size:1.1rem;color:#64748b;max-width:600px;margin:0 auto 32px;'>
        Hệ thống phân tích phản hồi sinh viên Trường Đại học Thủy Lợi<br>
        ứng dụng Trí tuệ Nhân tạo – Phân tích Cảm xúc Đa Khía cạnh (ABSA)
    </p>
</div>
""", unsafe_allow_html=True)

# ── Thẻ tính năng ─────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class='kpi-card' style='border-left-color:#4f46e5;text-align:center;'>
        <div style='font-size:2.5rem;margin-bottom:12px;'>📊</div>
        <div style='font-weight:700;color:#0f172a;font-size:1.1rem;margin-bottom:8px;'>Bảng điều khiển</div>
        <div style='color:#64748b;font-size:0.85rem;'>
            Tải lên file dữ liệu, chạy dự đoán và xem báo cáo tổng quan với biểu đồ, bộ lọc và tải xuống kết quả.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class='kpi-card' style='border-left-color:#10b981;text-align:center;'>
        <div style='font-size:2.5rem;margin-bottom:12px;'>⚡</div>
        <div style='font-weight:700;color:#0f172a;font-size:1.1rem;margin-bottom:8px;'>Dự đoán từng câu</div>
        <div style='color:#64748b;font-size:0.85rem;'>
            Nhập một câu tiếng Việt bất kỳ và nhận kết quả dự đoán cảm xúc theo từng khía cạnh ngay lập tức.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div class='gradient-divider'></div>", unsafe_allow_html=True)

# ── Cách hoạt động ────────────────────────────────────────────────────────────
st.markdown("""
<h2 style='text-align:center;color:#0f172a;font-weight:700;margin-bottom:24px;'>
    Hệ thống hoạt động như thế nào?
</h2>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
buoc = [
    ("1️⃣", "Nhập phản hồi",       "Sinh viên nhập câu tiếng Việt bất kỳ"),
    ("2️⃣", "Tách mệnh đề",        "AI tách theo từ nối: nhưng, tuy nhiên, còn…"),
    ("3️⃣", "Nhận diện khía cạnh", "Khớp từ khóa → Giảng viên, Cơ sở, Chương trình…"),
    ("4️⃣", "Dự đoán cảm xúc",     "Mô hình CNN + BiLSTM → Tích cực / Trung lập / Tiêu cực"),
]
for col, (icon, tieu_de, mo_ta) in zip([c1, c2, c3, c4], buoc):
    with col:
        st.markdown(f"""
        <div style='text-align:center;padding:16px;'>
            <div style='font-size:2rem;margin-bottom:8px;'>{icon}</div>
            <div style='font-weight:700;color:#0f172a;margin-bottom:6px;font-size:0.95rem;'>{tieu_de}</div>
            <div style='color:#64748b;font-size:0.8rem;'>{mo_ta}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center;'>
    <p style='color:#94a3b8;font-size:0.85rem;'>
        Chọn chức năng từ menu bên trái để bắt đầu →
        <strong style='color:#4f46e5;'>📊 Bảng điều khiển</strong> &nbsp;·&nbsp;
        <strong style='color:#10b981;'>⚡ Dự đoán từng câu</strong>
    </p>
</div>
""", unsafe_allow_html=True)
