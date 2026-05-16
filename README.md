# 🌊 TLU Feedback Analytics

Hệ thống phân tích phản hồi sinh viên Trường Đại học Thủy Lợi sử dụng AI – Aspect-Based Sentiment Analysis (ABSA).

## 🚀 Tính năng

- **Hybrid ABSA Pipeline**: Kết hợp rule-based clause splitting + CNN+BiLSTM model
- **Dashboard Analytics**: KPI cards, biểu đồ sentiment, word cloud, bảng dữ liệu có filter
- **Kiểm thử Từng câu**: Phân tích realtime, lịch sử dự đoán, câu mẫu gợi ý
- **Xử lý Hàng loạt**: Upload CSV/Excel, progress bar, export kết quả

## 🛠️ Cài đặt & Chạy local

```bash
# 1. Clone repository
git clone https://github.com/quangdungai/tlu-vsfc.git
cd tlu-vsfc

# 2. Tạo môi trường ảo (khuyến nghị)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# 3. Cài đặt thư viện
pip install -r requirements.txt

# 4. Train model (bỏ qua nếu đã có file weights)
python train.py

# 5. Chạy ứng dụng
streamlit run app.py
```

## 📁 Cấu trúc Project

```
tlu-vsfc/
├── app.py                           # Entry point – Trang chủ
├── train.py                         # Script huấn luyện model
├── pages/
│   ├── 1_📊_Dashboard.py           # Trang dashboard analytics
│   ├── 2_⚡_Predict.py             # Trang kiểm thử từng câu
│   └── 3_📂_Batch.py               # Trang xử lý hàng loạt
├── utils/
│   ├── absa_pipeline.py            # Hybrid ABSA pipeline
│   ├── model_loader.py             # Load model & tokenizer
│   ├── preprocessor.py             # Tiền xử lý tiếng Việt
│   └── visualizer.py              # Plotly charts
├── assets/
│   └── style.css                   # Global CSS
├── data/
│   ├── train/                      # Dữ liệu huấn luyện
│   ├── dev/                        # Dữ liệu validation
│   └── test/                       # Dữ liệu kiểm thử
├── sentiment_cnn_weights.weights.h5 # Weights của model đã train
├── tokenizer.pickle                 # Tokenizer đã fit
├── requirements.txt
└── README.md
```

## 🧠 Kiến trúc Model

- **Embedding Layer**: 10,000 vocab × 50 dims
- **Multi-kernel CNN**: Filters 128, kernels [3, 4, 5]
- **BiLSTM**: 64 units, bidirectional
- **Output**: 4 nhánh (Lecturer, Training, Facility, Others) × 4 classes (Neg/Neu/Pos/N/A)

## 🎯 ABSA Pipeline

```
Input → Clause Splitting → Keyword Matching → CNN+BiLSTM → Kết quả
"Cô dạy nhiệt tình nhưng phòng hơi nóng"
    → ["Cô dạy nhiệt tình", "phòng hơi nóng"]
    → [Lecturer, Facility]
    → [Positive, Negative]
```

## ☁️ Deploy

### Streamlit Cloud
1. Push code lên GitHub
2. Vào [share.streamlit.io](https://share.streamlit.io)
3. Connect repository, chọn `app.py`, deploy

### HuggingFace Spaces
1. Tạo Space mới, chọn SDK = Streamlit
2. Push code lên Space repository

## 📊 Dữ liệu

Bộ dữ liệu UIT-VSFC (Vietnamese Students' Feedback Corpus):
- ~16,000 câu phản hồi tiếng Việt
- 4 chủ đề: Lecturer, Training Program, Facility, Others
- 3 nhãn cảm xúc: Positive, Neutral, Negative
