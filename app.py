"""
app.py — Trang chủ TLU Feedback Analytics
"""
import streamlit as st
import os
import base64

st.set_page_config(
    page_title="TLU Feedback Analytics",
    page_icon="Logo-DH-Thuy-Loi.webp",
    layout="wide",
    initial_sidebar_state="expanded",
)

def load_css():
    css_path = os.path.join("assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

@st.cache_data
def get_base64_image(file_path):
    if not os.path.exists(file_path):
        return ""
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()



# ── Sidebar ───────────────────────────────────────────────────────────────────
_logo_b64 = get_base64_image("Logo-DH-Thuy-Loi.webp")
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

# ── Trang chủ ─────────────────────────────────────────────────────────────────
logo_base64 = get_base64_image("Logo-DH-Thuy-Loi.webp")
st.markdown(f"""
<div style='text-align:center;padding:60px 20px 40px;'>
    <div style='font-size:4rem;margin-bottom:16px;'>
        <img src='data:image/webp;base64,{logo_base64}' width='200'>
    </div>
    <h1 style='font-size:2.5rem;font-weight:800;color:#0f172a;margin-bottom:8px;'>
        TLU Feedback Analytics
    </h1>
    <p style='font-size:1.1rem;color:#64748b;max-width:600px;margin:0 auto 32px;'>
        Hệ thống phân tích phản hồi sinh viên Trường Đại học Thủy Lợi<br>
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
            Nhập một câu phản hồi tiếng Việt bất kỳ và nhận kết quả dự đoán cảm xúc theo từng khía cạnh ngay lập tức.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div style='text-align:center;margin-top:16px;'>
    <p style='color:#94a3b8;font-size:0.85rem;'>
        Chọn chức năng từ menu bên trái để bắt đầu →
        <strong style='color:#4f46e5;'>📊 Bảng điều khiển</strong> &nbsp;·&nbsp;
        <strong style='color:#10b981;'>⚡ Dự đoán từng câu</strong>
    </p>
</div>
""", unsafe_allow_html=True)
