"""
pages/3_📂_Dự đoán Hàng loạt.py
Tải file lên → Dự đoán → Lưu vào hệ thống → Dashboard tự cập nhật.
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import json
import time
import io

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_loader import get_model_and_tokenizer

st.set_page_config(page_title="Dự đoán Hàng loạt – TLU", page_icon="📂", layout="wide")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PERSIST_PATH = os.path.join(ROOT_DIR, "data", "current_data.csv")
META_PATH = os.path.join(ROOT_DIR, "data", "current_data_meta.json")

def load_css():
    css_path = os.path.join(ROOT_DIR, "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
load_css()

with st.sidebar:
    logo_path = os.path.join(ROOT_DIR, "Logo-DH-Thuy-Loi.webp")
    if os.path.exists(logo_path):
        st.image(logo_path, width=90)
    st.markdown("<p style='color:#a5b4fc;font-weight:700;'>TLU Analytics</p>", unsafe_allow_html=True)

model, tokenizer = get_model_and_tokenizer()
if model:
    st.sidebar.success("🟢 Mô hình đang hoạt động")
else:
    st.sidebar.error("🔴 Mô hình ngoại tuyến")

ASPECT_COLS = ['Lecturer_Sentiment', 'Training_Sentiment', 'Facility_Sentiment', 'Others_Sentiment']
ASPECT_VI = {
    'Lecturer_Sentiment': 'Giảng viên',
    'Training_Sentiment': 'Chương trình',
    'Facility_Sentiment': 'Cơ sở vật chất',
    'Others_Sentiment': 'Khác',
}

# ── Tiêu đề ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class='page-header'>
    <div style='font-size:2.5rem;'>📂</div>
    <div>
        <h1 class='page-title'>Dự đoán Hàng loạt</h1>
        <p class='page-subtitle'>Tải file lên, chạy dự đoán và lưu kết quả vào Bảng điều khiển</p>
    </div>
</div>
""", unsafe_allow_html=True)

if not model:
    st.error("❌ Không thể tải mô hình. Vui lòng kiểm tra file weights.")
    st.stop()

# ── Hàm dự đoán ──────────────────────────────────────────────────────────────
def du_doan_hang_loat(df: pd.DataFrame, cot_van_ban: str) -> pd.DataFrame:
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    texts = df[cot_van_ban].astype(str).tolist()
    seqs = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(seqs, maxlen=100, padding='post', truncating='post')
    preds = model.predict(padded, batch_size=64, verbose=0)
    nhan_cam_xuc = {0: "Tiêu cực 😠", 1: "Trung lập 😐", 2: "Tích cực 😄"}
    khia_canh = ['Lecturer', 'Training', 'Facility', 'Others']
    for i, kc in enumerate(khia_canh):
        lop_du_doan = np.argmax(preds[i], axis=1)
        nhan = [nhan_cam_xuc.get(cls) if cls < 3 else None for cls in lop_du_doan]
        df[f'{kc}_Sentiment'] = nhan
    return df

def luu_vao_disk(df: pd.DataFrame, ten_file: str):
    os.makedirs(os.path.dirname(PERSIST_PATH), exist_ok=True)
    df.to_csv(PERSIST_PATH, index=False, encoding="utf-8-sig")
    meta = {"source": ten_file, "rows": len(df)}
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)

def doc_meta() -> dict:
    if os.path.exists(META_PATH):
        try:
            with open(META_PATH, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

# ── Hiển thị trạng thái dữ liệu hiện tại ────────────────────────────────────
meta = doc_meta()
if meta:
    st.info(f"📌 Bảng điều khiển đang hiển thị: **{meta.get('source', 'N/A')}** — {meta.get('rows', 0):,} câu")

st.markdown("<div class='gradient-divider'></div>", unsafe_allow_html=True)

# ── Tải file ──────────────────────────────────────────────────────────────────
st.markdown("### 📤 Tải lên file phản hồi")
st.caption("Hỗ trợ định dạng: CSV, Excel (.xlsx), TXT (mỗi dòng một câu)")

uploaded_file = st.file_uploader(
    "Chọn file", type=["csv", "xlsx", "txt"],
    label_visibility="collapsed",
)

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df_raw = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            df_raw = pd.read_excel(uploaded_file)
        else:
            content = uploaded_file.read().decode("utf-8")
            df_raw = pd.DataFrame({"Phản hồi": [l.strip() for l in content.split("\n") if l.strip()]})
    except Exception as e:
        st.error(f"❌ Lỗi đọc file: {e}")
        st.stop()

    st.success(f"✅ Đã tải: **{uploaded_file.name}** — {len(df_raw):,} dòng")

    with st.expander("🔍 Xem trước dữ liệu (5 dòng đầu)", expanded=True):
        st.dataframe(df_raw.head(5), use_container_width=True)

    cot_van_ban = st.selectbox("📌 Chọn cột chứa câu phản hồi:", df_raw.columns)

    col_btn, col_note = st.columns([1, 3])
    with col_btn:
        chay_btn = st.button("▶ Bắt đầu dự đoán", type="primary", use_container_width=True)
    with col_note:
        st.markdown("<p style='color:#64748b;font-size:0.85rem;margin-top:10px;'>Sau khi dự đoán xong, kết quả sẽ được lưu và hiển thị trên Bảng điều khiển.</p>", unsafe_allow_html=True)

    if chay_btn:
        thanh_tien_trinh = st.progress(0, text="Đang chuẩn bị...")
        trang_thai = st.empty()

        kich_thuoc_block = 50
        df_lam_viec = df_raw.copy()
        n = len(df_lam_viec)
        cac_phan = []

        for bat_dau in range(0, n, kich_thuoc_block):
            ket_thuc = min(bat_dau + kich_thuoc_block, n)
            khoi = df_lam_viec.iloc[bat_dau:ket_thuc].copy()
            khoi_ket_qua = du_doan_hang_loat(khoi, cot_van_ban)
            cac_phan.append(khoi_ket_qua)
            phan_tram = int((ket_thuc / n) * 100)
            thanh_tien_trinh.progress(phan_tram, text=f"Đang dự đoán {ket_thuc}/{n} câu...")
            time.sleep(0.03)

        df_ket_qua = pd.concat(cac_phan, ignore_index=True)
        # Đổi tên cột về "Phản hồi" cho thống nhất
        if cot_van_ban in df_ket_qua.columns and cot_van_ban != "Phản hồi":
            df_ket_qua = df_ket_qua.rename(columns={cot_van_ban: "Phản hồi"})

        # Đổi tên cột sang tiếng Việt cho dashboard
        doi_ten = {
            'Lecturer_Sentiment': 'Giảng viên',
            'Training_Sentiment': 'Chương trình',
            'Facility_Sentiment': 'Cơ sở vật chất',
            'Others_Sentiment': 'Khác',
        }

        thanh_tien_trinh.progress(100, text="✅ Hoàn tất!")
        luu_vao_disk(df_ket_qua, uploaded_file.name)
        # Xóa cache dashboard để tải lại dữ liệu mới
        st.session_state.pop('dashboard_data', None)
        trang_thai.success(f"🎉 Dự đoán xong **{n:,}** câu! Kết quả đã được lưu vào Bảng điều khiển.")
        st.session_state['ket_qua_batch'] = df_ket_qua

# ── Bảng kết quả ─────────────────────────────────────────────────────────────
if 'ket_qua_batch' in st.session_state:
    df = st.session_state['ket_qua_batch']

    # Xác định cột sentiment hiện có
    cot_sentinel = [c for c in ASPECT_COLS if c in df.columns]

    st.markdown("<div class='gradient-divider'></div>", unsafe_allow_html=True)
    st.markdown("### 📄 Kết quả dự đoán")

    def to_mau(val):
        if pd.isna(val): return ''
        if "Tiêu cực" in str(val): return 'background-color:#fee2e2;color:#991b1b;font-weight:600'
        if "Tích cực" in str(val): return 'background-color:#dcfce7;color:#166534;font-weight:600'
        if "Trung lập" in str(val): return 'background-color:#fef9c3;color:#854d0e;font-weight:600'
        return ''

    # Đổi tên cột hiển thị sang tiếng Việt
    df_hien_thi = df.copy()
    df_hien_thi = df_hien_thi.rename(columns={
        'Lecturer_Sentiment': 'Giảng viên',
        'Training_Sentiment': 'Chương trình đào tạo',
        'Facility_Sentiment': 'Cơ sở vật chất',
        'Others_Sentiment': 'Khác',
    })
    cot_mau = [c for c in ['Giảng viên', 'Chương trình đào tạo', 'Cơ sở vật chất', 'Khác'] if c in df_hien_thi.columns]

    st.dataframe(
        df_hien_thi.style.map(to_mau, subset=cot_mau),
        use_container_width=True,
        height=430,
    )

    # Tải xuống
    col1, col2 = st.columns(2)
    with col1:
        csv_bytes = df_hien_thi.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "📥 Tải xuống CSV",
            data=csv_bytes,
            file_name="ket_qua_du_doan.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col2:
        try:
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine='openpyxl') as writer:
                df_hien_thi.to_excel(writer, index=False, sheet_name='Kết quả')
            st.download_button(
                "📥 Tải xuống Excel",
                data=buf.getvalue(),
                file_name="ket_qua_du_doan.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        except Exception:
            pass
