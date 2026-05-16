"""
pages/1_📊_Bảng điều khiển.py
Tải file lên → Dự đoán → Xem biểu đồ & kết quả — tất cả trong một trang.
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
from utils.visualizer import sentiment_donut, aspect_bar_chart
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(page_title="Bảng điều khiển – TLU", page_icon="📊", layout="wide")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PERSIST_PATH = os.path.join(ROOT_DIR, "data", "current_data.csv")
META_PATH   = os.path.join(ROOT_DIR, "data", "current_data_meta.json")

# ── CSS ───────────────────────────────────────────────────────────────────────
def load_css():
    css_path = os.path.join(ROOT_DIR, "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
load_css()

# ── Sidebar ───────────────────────────────────────────────────────────────────
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

# ── Hằng số ───────────────────────────────────────────────────────────────────
ASPECT_COLS = ['Lecturer_Sentiment', 'Training_Sentiment', 'Facility_Sentiment', 'Others_Sentiment']
ASPECT_DISPLAY = {
    'Lecturer_Sentiment': 'Giảng viên',
    'Training_Sentiment': 'Chương trình',
    'Facility_Sentiment': 'Cơ sở vật chất',
    'Others_Sentiment': 'Khác',
}

# ── Tiêu đề ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class='page-header'>
    <div style='font-size:2.5rem;'>📊</div>
    <div>
        <h1 class='page-title'>Bảng điều khiển</h1>
        <p class='page-subtitle'>Tải lên dữ liệu, chạy dự đoán và xem báo cáo tổng quan</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Hàm tiện ích ──────────────────────────────────────────────────────────────
def tim_cot_van_ban(df: pd.DataFrame) -> str:
    for ten in ['Phản hồi', 'Feedback', 'text', 'sentence', 'comment', 'review']:
        if ten in df.columns:
            return ten
    khong_phai_sentiment = [c for c in df.columns if c not in ASPECT_COLS]
    return khong_phai_sentiment[0] if khong_phai_sentiment else df.columns[0]

def du_doan_hang_loat(df: pd.DataFrame, cot: str) -> pd.DataFrame:
    if not model or not tokenizer:
        return df
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    texts = df[cot].astype(str).tolist()
    seqs  = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(seqs, maxlen=100, padding='post', truncating='post')
    preds  = model.predict(padded, batch_size=64, verbose=0)
    nhan   = {0: "Tiêu cực 😠", 1: "Trung lập 😐", 2: "Tích cực 😄"}
    for i, kc in enumerate(['Lecturer', 'Training', 'Facility', 'Others']):
        lop = np.argmax(preds[i], axis=1)
        df[f'{kc}_Sentiment'] = [nhan.get(c) if c < 3 else None for c in lop]
    return df

def luu_disk(df: pd.DataFrame, ten_file: str):
    os.makedirs(os.path.dirname(PERSIST_PATH), exist_ok=True)
    df.to_csv(PERSIST_PATH, index=False, encoding="utf-8-sig")
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump({"source": ten_file, "rows": len(df)}, f, ensure_ascii=False)

def doc_disk() -> pd.DataFrame | None:
    if os.path.exists(PERSIST_PATH):
        try:
            return pd.read_csv(PERSIST_PATH, encoding="utf-8-sig")
        except Exception:
            return None
    return None

def doc_meta() -> dict:
    if os.path.exists(META_PATH):
        try:
            with open(META_PATH, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

@st.cache_data
def tai_du_lieu_mau():
    try:
        with open(os.path.join(ROOT_DIR, "data", "test", "sents.txt"), "r", encoding="utf-8") as f:
            lines = f.readlines()
        df = pd.DataFrame({"Phản hồi": [l.strip() for l in lines[:500] if l.strip()]})
        return du_doan_hang_loat(df, "Phản hồi")
    except Exception:
        return None

def to_mau(val):
    if pd.isna(val): return ''
    if "Tiêu cực" in str(val): return 'background-color:#fee2e2;color:#991b1b;font-weight:600'
    if "Tích cực"  in str(val): return 'background-color:#dcfce7;color:#166534;font-weight:600'
    if "Trung lập" in str(val): return 'background-color:#fef9c3;color:#854d0e;font-weight:600'
    return ''

def y_kien_tieu_bieu(sentences, top_n=5):
    sentences = [str(s) for s in sentences if str(s).strip()]
    if len(sentences) <= top_n:
        return sentences
    try:
        vec = TfidfVectorizer(max_df=0.85)
        X   = vec.fit_transform(sentences)
        scores = np.array(X.sum(axis=1)).flatten()
        return [sentences[i] for i in scores.argsort()[-top_n:][::-1]]
    except Exception:
        return sentences[:top_n]

# ── Khu vực tải file ──────────────────────────────────────────────────────────
meta = doc_meta()
# if meta:
#     st.info(
#         f"📌 Đang hiển thị: **{meta.get('source','N/A')}** — {meta.get('rows',0):,} câu  "
#         f"| Tải file mới bên dưới để thay thế."
#     )

with st.expander("📂 Tải lên file phản hồi mới", expanded=not bool(meta)):
    uploaded = st.file_uploader(
        "Hỗ trợ: CSV, Excel (.xlsx), TXT (mỗi dòng một câu)",
        type=["csv", "xlsx", "txt"],
        label_visibility="visible",
    )

    if uploaded:
        try:
            if uploaded.name.endswith(".csv"):
                df_raw = pd.read_csv(uploaded)
            elif uploaded.name.endswith(".xlsx"):
                df_raw = pd.read_excel(uploaded)
            else:
                content = uploaded.read().decode("utf-8")
                df_raw  = pd.DataFrame({"Phản hồi": [l.strip() for l in content.split("\n") if l.strip()]})
        except Exception as e:
            st.error(f"❌ Lỗi đọc file: {e}")
            df_raw = None

        if df_raw is not None:
            st.success(f"✅ Đã tải: **{uploaded.name}** — {len(df_raw):,} dòng")
            with st.expander("🔍 Xem trước (5 dòng đầu)"):
                st.dataframe(df_raw.head(5), use_container_width=True)

            cot = st.selectbox("📌 Chọn cột chứa câu phản hồi:", df_raw.columns)

            col_btn, col_hint = st.columns([1, 4])
            with col_btn:
                bat_dau = st.button("▶ Bắt đầu dự đoán", type="primary", use_container_width=True)
            with col_hint:
                st.caption("Kết quả sẽ được lưu lại và hiển thị ngay bên dưới.")

            if bat_dau:
                if not model:
                    st.error("❌ Mô hình chưa được tải. Kiểm tra file weights.")
                else:
                    thanh = st.progress(0, text="Đang chuẩn bị...")
                    n, buoc, cac_phan = len(df_raw), 50, []
                    for bd in range(0, n, buoc):
                        kt   = min(bd + buoc, n)
                        khoi = df_raw.iloc[bd:kt].copy()
                        cac_phan.append(du_doan_hang_loat(khoi, cot))
                        thanh.progress(int(kt / n * 100), text=f"Đang dự đoán {kt}/{n} câu…")
                        time.sleep(0.03)
                    df_kq = pd.concat(cac_phan, ignore_index=True)
                    if cot not in ["Phản hồi", "Feedback"]:
                        df_kq = df_kq.rename(columns={cot: "Phản hồi"})
                    thanh.progress(100, text="✅ Hoàn tất!")
                    luu_disk(df_kq, uploaded.name)
                    st.session_state.pop('du_lieu', None)   # invalidate cache
                    st.success(f"🎉 Dự đoán xong **{n:,}** câu! Bảng điều khiển đã cập nhật.")
                    st.rerun()

    # Nút xóa về dữ liệu mẫu
    if meta:
        if st.button("🗑️ Xóa & dùng lại dữ liệu mẫu", key="xoa_disk"):
            for p in [PERSIST_PATH, META_PATH]:
                if os.path.exists(p):
                    os.remove(p)
            st.session_state.pop('du_lieu', None)
            st.rerun()

st.markdown("<div class='gradient-divider'></div>", unsafe_allow_html=True)

# ── Tải dữ liệu (ưu tiên: session → disk → mẫu) ──────────────────────────────
if 'du_lieu' not in st.session_state:
    disk = doc_disk()
    if disk is not None:
        st.session_state['du_lieu'] = disk
    else:
        with st.spinner("Đang tải dữ liệu mẫu…"):
            st.session_state['du_lieu'] = tai_du_lieu_mau()

df = st.session_state.get('du_lieu')

if df is None or df.empty:
    st.warning("⚠️ Chưa có dữ liệu. Hãy tải file lên ở trên.")
    st.stop()

# ── KPI ───────────────────────────────────────────────────────────────────────
tat_ca = df[ASPECT_COLS].melt().dropna()
tong   = len(tat_ca)
pct_pos = (tat_ca['value'] == "Tích cực 😄").sum() / tong * 100 if tong else 0
pct_neg = (tat_ca['value'] == "Tiêu cực 😠").sum() / tong * 100 if tong else 0
so_nhac = {c: df[c].notna().sum() for c in ASPECT_COLS}
top_col = max(so_nhac, key=so_nhac.get)
top_aspect = ASPECT_DISPLAY.get(top_col, top_col)

k1, k2, k3, k4 = st.columns(4)
for col, color, icon, tieu_de, gia_tri, ghi_chu in [
    (k1, "#4f46e5", "📝", "Tổng phản hồi",  f"{len(df):,}",        "câu đã dự đoán"),
    (k2, "#10b981", "😄", "Tích cực",        f"{pct_pos:.1f}%",     "trên tất cả khía cạnh"),
    (k3, "#ef4444", "😠", "Tiêu cực",        f"{pct_neg:.1f}%",     "cần cải thiện"),
    (k4, "#3b82f6", "🔥", "Khía cạnh nổi bật", top_aspect,         "được nhắc đến nhiều nhất"),
]:
    with col:
        st.markdown(f"""
        <div class='kpi-card' style='border-left-color:{color};'>
            <div class='kpi-icon'>{icon}</div>
            <div class='kpi-title'>{tieu_de}</div>
            <div class='kpi-value'>{gia_tri}</div>
            <div class='kpi-sub'>{ghi_chu}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📈 Tổng quan", "🗂️ Chi tiết & Lọc"])

with tab1:
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("<div class='chart-box'>", unsafe_allow_html=True)
        st.plotly_chart(sentiment_donut(tat_ca['value'], "Tổng quan cảm xúc"), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='chart-box'>", unsafe_allow_html=True)
        st.plotly_chart(aspect_bar_chart(df, ASPECT_COLS), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Word cloud
    try:
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
        cot_vb = tim_cot_van_ban(df)
        all_text = " ".join(df[cot_vb].astype(str).tolist())
        wc = WordCloud(width=900, height=280, background_color="white",
                       colormap="RdYlGn", max_words=100).generate(all_text)
        fig_wc, ax = plt.subplots(figsize=(12, 3.5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        fig_wc.patch.set_alpha(0)
        st.markdown("<div class='chart-box'>", unsafe_allow_html=True)
        st.markdown("#### ☁️ Word cloud phản hồi")
        st.pyplot(fig_wc)
        st.markdown("</div>", unsafe_allow_html=True)
    except ImportError:
        pass

with tab2:
    st.markdown("<div class='chart-box'>", unsafe_allow_html=True)

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        chon_kc = st.selectbox("Lọc theo khía cạnh", ["Tất cả"] + list(ASPECT_DISPLAY.values()))
    with col_f2:
        chon_cx = st.selectbox("Lọc theo cảm xúc", ["Tất cả", "Tích cực 😄", "Tiêu cực 😠", "Trung lập 😐"])

    df_loc = df.copy()
    anh_xa = {v: k for k, v in ASPECT_DISPLAY.items()}
    if chon_kc != "Tất cả":
        col_kc = anh_xa.get(chon_kc, "")
        if col_kc:
            df_loc = df_loc[df_loc[col_kc] == chon_cx] if chon_cx != "Tất cả" else df_loc[df_loc[col_kc].notna()]
    elif chon_cx != "Tất cả":
        mask = pd.Series(False, index=df_loc.index)
        for c in ASPECT_COLS:
            mask |= (df_loc[c] == chon_cx)
        df_loc = df_loc[mask]

    st.markdown(f"<p style='color:#64748b;font-size:0.85rem;'>Hiển thị <strong>{len(df_loc):,}</strong> câu</p>",
                unsafe_allow_html=True)

    # Ý kiến tiêu biểu
    st.markdown("---")
    st.markdown("##### 💡 Ý kiến tiêu biểu")
    cot_vb = tim_cot_van_ban(df_loc)
    for s in y_kien_tieu_bieu(df_loc[cot_vb].tolist(), top_n=5):
        st.markdown(f"> *\"{s}\"*")

    st.markdown("---")
    st.markdown("##### 📄 Bảng kết quả")

    # Đổi tên cột hiển thị sang tiếng Việt
    df_hien_thi = df_loc.rename(columns={
        'Lecturer_Sentiment': 'Giảng viên',
        'Training_Sentiment': 'Chương trình',
        'Facility_Sentiment': 'Cơ sở vật chất',
        'Others_Sentiment':   'Khác',
    })
    cot_mau = [c for c in ['Giảng viên', 'Chương trình', 'Cơ sở vật chất', 'Khác'] if c in df_hien_thi.columns]
    st.dataframe(df_hien_thi.style.map(to_mau, subset=cot_mau), use_container_width=True, height=420)

    # Tải xuống
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            "📥 Tải xuống CSV",
            data=df_hien_thi.to_csv(index=False).encode("utf-8-sig"),
            file_name="ket_qua_du_doan.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col_dl2:
        try:
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as w:
                df_hien_thi.to_excel(w, index=False, sheet_name="Kết quả")
            st.download_button(
                "📥 Tải xuống Excel",
                data=buf.getvalue(),
                file_name="ket_qua_du_doan.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        except Exception:
            pass

    st.markdown("</div>", unsafe_allow_html=True)
