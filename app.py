import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer

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

# Setup Page Config
st.set_page_config(
    page_title="TLU Feedback Analytics", 
    page_icon="🌊", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- CSS Styling for Realistic Admin Dashboard ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .main {
        background-color: #f3f4f6;
    }
    .kpi-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border-left: 5px solid #4f46e5;
        margin-bottom: 20px;
        height: 100%;
    }
    .kpi-title {
        color: #6b7280;
        font-size: 0.875rem;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 5px;
    }
    .kpi-value {
        color: #111827;
        font-size: 1.5rem;
        font-weight: 800;
    }
    .kpi-sub {
        color: #10b981;
        font-size: 0.875rem;
        font-weight: 600;
        margin-top: 5px;
    }
    .dashboard-header {
        font-size: 2rem;
        font-weight: 800;
        color: #1f2937;
        margin-bottom: 0px;
        padding-bottom: 0px;
    }
    .dashboard-subtitle {
        color: #6b7280;
        margin-bottom: 30px;
    }
    .chart-container {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- Constants & Resources ---
MAX_SEQUENCE_LENGTH = 100
MODEL_PATH = "sentiment_cnn_model.keras"
TOKENIZER_PATH = "tokenizer.pickle"

@st.cache_resource(show_spinner="Loading ABSA Neural Network...")
def load_resources():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(TOKENIZER_PATH):
        return None, None
    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, 'rb') as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer

model, tokenizer = load_resources()

# Mappings
sentiment_map = {0: "Negative 😠", 1: "Neutral 😐", 2: "Positive 😄"}
sentiment_colors = {0: "#ef4444", 1: "#eab308", 2: "#10b981"}
aspect_names = ["Lecturer 👨‍🏫", "Training Program 📚", "Facility 🏢", "Others 🔄"]

# Helper functions
def predict_text(text):
    if not model or not tokenizer:
        return {}
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    preds = model.predict(padded, verbose=0)
    
    results = {}
    for i, aspect in enumerate(aspect_names):
        pred = preds[i][0]
        sentiment_class = int(np.argmax(pred))
        confidence = float(np.max(pred))
        
        # 3 means Not Mentioned
        if sentiment_class < 3:
            results[aspect] = {
                "class": sentiment_class,
                "confidence": confidence,
                "distribution": pred[:3] # only pass Neg/Neu/Pos probs
            }
            
    return results

def predict_batch(df, text_col):
    if not model or not tokenizer:
        return df
    
    texts = df[text_col].astype(str).tolist()
    seqs = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(seqs, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    
    preds = model.predict(padded, batch_size=64, verbose=0)
    aspect_keys = ['Lecturer', 'Training', 'Facility', 'Others']
    
    for i, aspect in enumerate(aspect_keys):
        pred_classes = np.argmax(preds[i], axis=1)
        labels = [sentiment_map.get(cls) if cls < 3 else None for cls in pred_classes]
        df[f'{aspect}_Sentiment'] = labels
        
    return df

@st.cache_data
def load_demo_data():
    try:
        with open("data/test/sents.txt", "r", encoding="utf-8") as f:
            lines = f.readlines()
        df = pd.DataFrame({"Feedback": [line.strip() for line in lines[:500]]})
        return predict_batch(df, "Feedback")
    except:
        return None

# --- Sidebar Navigation ---
with st.sidebar:
    st.image("Logo-DH-Thuy-Loi.webp", width=100)
    st.markdown("## TLU Analytics")
    st.markdown("<br>", unsafe_allow_html=True)
    
    page = st.radio("Menu", ["📊 Dashboard Phân tích", "⚡ Kiểm thử Từng câu"])
    
    st.markdown("---")
    st.markdown("### System Status")
    if model:
        st.success("🟢 ABSA Model Online")
    else:
        st.error("🔴 Model Offline")

# --- Pages ---

if page == "📊 Dashboard Phân tích":
    col1, col2 = st.columns([1, 15])
    with col1:
        st.image("Logo-DH-Thuy-Loi.webp", width=60)
    with col2:
        st.markdown("<h1 class='dashboard-header'>Phân tích Phản hồi TLU</h1>", unsafe_allow_html=True)
    st.markdown("<p class='dashboard-subtitle'>Hệ thống phân tích đa khía cạnh đánh giá của sinh viên Trường Đại học Thủy Lợi.</p>", unsafe_allow_html=True)
    
    # --- File Upload Section (Moved from Batch Processing) ---
    with st.expander("📂 Tải lên Dữ liệu mới (Xử lý hàng loạt)", expanded=False):
        uploaded_file = st.file_uploader("Kéo thả file CSV, XLSX, hoặc TXT vào đây", type=["csv", "xlsx", "txt"])
        if uploaded_file is not None:
            if uploaded_file.name.endswith(".csv"):
                df_upload = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(".xlsx"):
                df_upload = pd.read_excel(uploaded_file)
            else:
                content = uploaded_file.read().decode("utf-8")
                lines = content.split("\n")
                df_upload = pd.DataFrame({"Feedback": [line.strip() for line in lines if line.strip()]})
                
            text_col = st.selectbox("Chọn cột chứa câu nhận xét:", df_upload.columns)
            
            if st.button("Phân tích AI 🚀", type="primary"):
                with st.spinner("Hệ thống đang chạy ABSA Pipeline..."):
                    results_df = predict_batch(df_upload, text_col)
                    if text_col != "Feedback":
                        results_df.rename(columns={text_col: "Feedback"}, inplace=True)
                        
                    st.session_state['dashboard_data'] = results_df
                    st.success("✅ Phân tích hoàn tất! Bảng điều khiển đã được cập nhật bên dưới.")

    st.markdown("---")
    
    # --- Analytics Dashboard ---
    if 'dashboard_data' not in st.session_state:
        with st.spinner("Đang tải dữ liệu mẫu..."):
            st.session_state['dashboard_data'] = load_demo_data()
            
    df = st.session_state.get('dashboard_data')
    
    if df is not None and not df.empty:
        # Calculate overall metrics
        aspect_cols = ['Lecturer_Sentiment', 'Training_Sentiment', 'Facility_Sentiment', 'Others_Sentiment']
        all_sentiments = df[aspect_cols].melt().dropna()
        
        pos_count = (all_sentiments['value'] == "Positive 😄").sum()
        neg_count = (all_sentiments['value'] == "Negative 😠").sum()
        total_mentions = len(all_sentiments)
        
        pos_pct = (pos_count / total_mentions * 100) if total_mentions > 0 else 0
        neg_pct = (neg_count / total_mentions * 100) if total_mentions > 0 else 0
        
        # Find most discussed aspect
        mention_counts = {col.split('_')[0]: df[col].notna().sum() for col in aspect_cols}
        top_aspect = max(mention_counts, key=mention_counts.get)
        
        tab1, tab2 = st.tabs(["📈 Báo cáo Tổng quan", "🗂️ Phân tích Chi tiết"])
        
        with tab1:
            st.markdown("<br>", unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-title">Total Reviews</div>
                    <div class="kpi-value">{len(df):,}</div>
                    <div class="kpi-sub">Processed texts</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="kpi-card" style="border-left-color: #10b981;">
                    <div class="kpi-title">Positive Mentions</div>
                    <div class="kpi-value">{pos_pct:.1f}%</div>
                    <div class="kpi-sub" style="color:#10b981;">Across all aspects</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="kpi-card" style="border-left-color: #ef4444;">
                    <div class="kpi-title">Negative Mentions</div>
                    <div class="kpi-value">{neg_pct:.1f}%</div>
                    <div class="kpi-sub" style="color:#ef4444;">Needs Action</div>
                </div>
                """, unsafe_allow_html=True)
            with col4:
                st.markdown(f"""
                <div class="kpi-card" style="border-left-color: #3b82f6;">
                    <div class="kpi-title">Top Aspect</div>
                    <div class="kpi-value">{top_aspect}</div>
                    <div class="kpi-sub" style="color:#6b7280;">Most discussed</div>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("<br>", unsafe_allow_html=True)
            c1, c2 = st.columns([1, 2])
            
            with c1:
                st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
                st.markdown("#### Overall Sentiment")
                sent_counts = all_sentiments['value'].value_counts().reset_index()
                fig_pie = px.pie(sent_counts, values='count', names='value', hole=0.6,
                                 color='value', 
                                 color_discrete_map={"Negative 😠": "#ef4444", "Neutral 😐": "#eab308", "Positive 😄": "#10b981"})
                fig_pie.update_layout(margin=dict(t=10, b=10, l=10, r=10), showlegend=False, height=350)
                st.plotly_chart(fig_pie, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
            with c2:
                st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
                st.markdown("#### Sentiment Breakdown by Aspect")
                # Grouped bar chart
                bar_data = []
                for col in aspect_cols:
                    aspect_name = col.split('_')[0]
                    counts = df[col].value_counts().to_dict()
                    for sent, count in counts.items():
                        bar_data.append({'Aspect': aspect_name, 'Sentiment': sent, 'Count': count})
                        
                df_bar = pd.DataFrame(bar_data)
                fig_bar = px.bar(df_bar, x='Aspect', y='Count', color='Sentiment', barmode='group',
                                 color_discrete_map={"Negative 😠": "#ef4444", "Neutral 😐": "#eab308", "Positive 😄": "#10b981"})
                fig_bar.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=350, 
                                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig_bar, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

        with tab2:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
            st.markdown("#### Lọc & Trích xuất Dữ liệu")
            
            filter_col1, filter_col2 = st.columns(2)
            with filter_col1:
                selected_aspect = st.selectbox("Lọc theo Khía cạnh", ["Tất cả"] + aspect_names)
            with filter_col2:
                selected_sentiment = st.selectbox("Lọc theo Cảm xúc", ["Tất cả", "Negative 😠", "Neutral 😐", "Positive 😄"])
                
            filtered_df = df.copy()
            if selected_aspect != "Tất cả":
                col_name = selected_aspect.split(" ")[0] + "_Sentiment"
                if selected_sentiment != "Tất cả":
                    filtered_df = filtered_df[filtered_df[col_name] == selected_sentiment]
                else:
                    filtered_df = filtered_df[filtered_df[col_name].notna()]
            else:
                if selected_sentiment != "Tất cả":
                    mask = False
                    for c in aspect_cols:
                        mask = mask | (filtered_df[c] == selected_sentiment)
                    filtered_df = filtered_df[mask]

            st.markdown("---")
            st.markdown("##### 💡 Tổng hợp Ý kiến Tiêu biểu")
            st.markdown("<p style='color: #6b7280; font-size: 0.9rem;'>Hệ thống AI tự động trích xuất các nhận xét mang tính đại diện cao nhất.</p>", unsafe_allow_html=True)
            
            texts_to_summarize = filtered_df['Feedback'].tolist()
            top_sentences = get_representative_sentences(texts_to_summarize, top_n=5)
            
            if top_sentences:
                for idx, sent in enumerate(top_sentences):
                    st.markdown(f"> *\"{sent}\"*")
            else:
                st.info("Chưa có ý kiến nào để tổng hợp.")
                
            st.markdown("---")
            st.markdown("##### 📄 Bảng Dữ liệu Chi tiết")

            def color_sentiment(val):
                if pd.isna(val):
                    return ''
                if "Negative" in str(val):
                    return 'background-color: #fecaca; color: #991b1b; font-weight: bold'
                elif "Positive" in str(val):
                    return 'background-color: #bbf7d0; color: #166534; font-weight: bold'
                elif "Neutral" in str(val):
                    return 'background-color: #fef08a; color: #854d0e; font-weight: bold'
                return ''

            st.dataframe(filtered_df.style.map(color_sentiment, subset=aspect_cols), use_container_width=True, height=400)
            
            # Download button placed at the end of the dashboard
            csv = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Xuất dữ liệu đã lọc (CSV)", data=csv, file_name="tlu_absa_filtered.csv", mime="text/csv")
            st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.warning("Chưa có dữ liệu.")

elif page == "⚡ Kiểm thử Từng câu":
    col1, col2 = st.columns([1, 15])
    with col1:
        st.image("Logo-DH-Thuy-Loi.webp", width=60)
    with col2:
        st.markdown("<h1 class='dashboard-header'>Kiểm thử Mô hình TLU</h1>", unsafe_allow_html=True)
    st.markdown("<p class='dashboard-subtitle'>Kiểm tra khả năng phân tách cảm xúc của sinh viên Thủy Lợi theo nhiều khía cạnh cùng lúc.</p>", unsafe_allow_html=True)
    
    if not model:
        st.error("Model not loaded.")
    else:
        with st.form("predict_form"):
            text_input = st.text_area("Nhập nhận xét của sinh viên (VD: 'Cô dạy nhiệt tình nhưng phòng hơi nóng'):", height=100)
            submitted = st.form_submit_button("Phân tích Đa Khía Cạnh 🚀", type="primary")
            
        if submitted and text_input.strip():
            with st.spinner("Extracting aspects..."):
                results = predict_text(text_input)
                
                st.markdown("<br>", unsafe_allow_html=True)
                if not results:
                    st.info("The model did not detect any specific topics in this sentence.")
                else:
                    # Dynamically create columns based on how many aspects were detected
                    cols = st.columns(len(results))
                    
                    for i, (aspect, data) in enumerate(results.items()):
                        cls = data['class']
                        color = sentiment_colors.get(cls, '#000')
                        
                        with cols[i]:
                            st.markdown(f"""
                            <div class="kpi-card" style="border-left-color: {color};">
                                <div class="kpi-title">{aspect}</div>
                                <div class="kpi-value" style="color: {color}; font-size: 1.2rem;">{sentiment_map.get(cls)}</div>
                                <div class="kpi-sub" style="color: #6b7280;">Conf: {data['confidence']:.2%}</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            df_dist = pd.DataFrame({
                                "Sentiment": ["Negative", "Neutral", "Positive"],
                                "Prob": data['distribution']
                            })
                            fig = px.bar(df_dist, x="Prob", y="Sentiment", orientation='h',
                                       color="Sentiment", color_discrete_sequence=["#ef4444", "#eab308", "#10b981"])
                            fig.update_layout(height=150, margin=dict(l=0, r=0, t=10, b=0), showlegend=False, xaxis_title="", yaxis_title="")
                            st.plotly_chart(fig, use_container_width=True)


