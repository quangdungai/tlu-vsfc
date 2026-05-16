"""
absa_pipeline.py
Hybrid ABSA Pipeline: Rule-based clause splitting + CNN model inference.
Chiến lược: tách câu dài → nhiều clause ngắn → detect aspect + predict sentiment.
"""
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils.preprocessor import clean_text, split_clauses

MAX_SEQUENCE_LENGTH = 100

# ── Aspect Keyword Map ─────────────────────────────────────────────────────────
ASPECT_KEYWORDS = {
    "Giảng viên 👨‍🏫": [
        "thầy", "cô", "giáo viên", "giảng viên", "dạy", "giảng", "hướng dẫn",
        "giáo", "thầy giáo", "cô giáo", "gv", "giảng dạy", "giải thích",
        "nhiệt tình", "tận tâm", "tận tình", "vô duyên", "xàm", "khó tính"
    ],
    "Cơ sở vật chất 🏢": [
        "phòng", "máy tính", "điều hòa", "bảng", "ghế", "bàn", "màn hình",
        "wifi", "mạng", "internet", "cơ sở vật chất", "csvc", "nóng", "lạnh",
        "sáng", "tối", "thiết bị", "máy chiếu", "âm thanh", "micro", "loa",
        "phòng học", "phòng máy", "phòng thực hành", "ký túc xá"
    ],
    "Chương trình đào tạo 📚": [
        "chương trình", "môn học", "giáo trình", "tài liệu", "đào tạo",
        "học phần", "tín chỉ", "lý thuyết", "thực hành", "thực tế", "thực tiễn",
        "kiến thức", "kỹ năng", "khó", "dễ", "nhàm", "chán", "hấp dẫn",
        "bổ ích", "thiết thực", "môn", "bài học", "syllabus", "nội dung"
    ],
    "Dịch vụ & Hành chính 🗂️": [
        "dịch vụ", "hỗ trợ", "văn phòng", "thủ tục", "hành chính",
        "sinh viên vụ", "phòng đào tạo", "cổng thông tin", "website",
        "học phí", "học bổng", "thư viện", "căng tin", "quán ăn"
    ],
}

# Nhãn sentiment từ model
SENTIMENT_MAP = {0: "Tiêu cực 😠", 1: "Trung lập 😐", 2: "Tích cực 😄"}
SENTIMENT_COLORS = {0: "#ef4444", 1: "#eab308", 2: "#10b981"}
SENTIMENT_EN = {0: "Negative", 1: "Neutral", 2: "Positive"}


def detect_aspect(clause: str) -> str | None:
    """
    Phát hiện khía cạnh trong một clause bằng keyword matching.
    Trả về aspect name hoặc None nếu không tìm được.
    """
    clause_lower = clause.lower()
    # Đếm số keyword match cho mỗi aspect
    scores = {}
    for aspect, keywords in ASPECT_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in clause_lower)
        if score > 0:
            scores[aspect] = score

    if not scores:
        return None
    # Trả về aspect có điểm cao nhất
    return max(scores, key=scores.get)


def predict_sentiment(text: str, model, tokenizer) -> dict:
    """
    Dùng CNN+BiLSTM model để predict sentiment trên toàn câu.
    Trả về dict {class, label, confidence, distribution}.
    """
    seq = tokenizer.texts_to_sequences([clean_text(text)])
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    preds = model.predict(padded, verbose=0)

    # Tổng hợp từ 4 output branches → lấy branch có confidence cao nhất (không phải N/A)
    best_cls = 1  # default neutral
    best_conf = 0.0
    best_dist = None

    for branch_pred in preds:
        prob = branch_pred[0][:3]  # chỉ lấy 3 class Neg/Neu/Pos
        cls = int(np.argmax(prob))
        conf = float(np.max(prob))
        if conf > best_conf:
            best_conf = conf
            best_cls = cls
            best_dist = prob

    return {
        "class": best_cls,
        "label": SENTIMENT_MAP[best_cls],
        "color": SENTIMENT_COLORS[best_cls],
        "confidence": best_conf,
        "distribution": best_dist.tolist() if best_dist is not None else [0.33, 0.33, 0.34],
    }


def predict_aspect_sentiment(text: str, model, tokenizer) -> dict:
    """
    Dùng toàn bộ 4 output branches của model để predict từng aspect.
    Trả về dict {aspect_key: {class, label, confidence, distribution}}.
    """
    aspect_keys = ["Giảng viên 👨‍🏫", "Chương trình đào tạo 📚", "Cơ sở vật chất 🏢", "Khác 🔄"]
    seq = tokenizer.texts_to_sequences([clean_text(text)])
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    preds = model.predict(padded, verbose=0)

    results = {}
    for i, aspect in enumerate(aspect_keys):
        prob = preds[i][0]
        cls = int(np.argmax(prob))
        conf = float(np.max(prob))
        if cls < 3:  # Loại bỏ class 3 = Not Mentioned
            results[aspect] = {
                "class": cls,
                "label": SENTIMENT_MAP[cls],
                "color": SENTIMENT_COLORS[cls],
                "confidence": conf,
                "distribution": prob[:3].tolist(),
            }
    return results


def run_absa_pipeline(text: str, model, tokenizer) -> list[dict]:
    """
    Pipeline ABSA hoàn chỉnh (Hybrid: Rule-based + Model).

    Bước 1: Tách câu → clauses (theo từ nối: nhưng, tuy nhiên...)
    Bước 2: Mỗi clause → detect aspect bằng keyword matching
    Bước 3: Mỗi clause → predict sentiment bằng CNN model
    Bước 4: Nếu không tách được clause, fallback về model branches
    Bước 5: Bổ sung aspect còn thiếu từ model branches (không duplicate)

    Trả về list[dict]:
      [{"aspect": str, "clause": str, "sentiment": dict}, ...]
    """
    if not model or not tokenizer:
        return []

    clauses = split_clauses(clean_text(text))
    results = []
    detected_aspects = set()

    # ── Phase 1: Hybrid (clause split + keyword matching) ─────────────────
    if len(clauses) > 1:
        for clause in clauses:
            aspect = detect_aspect(clause)
            if aspect and aspect not in detected_aspects:
                sentiment = predict_sentiment(clause, model, tokenizer)
                results.append({
                    "aspect": aspect,
                    "clause": clause,
                    "sentiment": sentiment,
                })
                detected_aspects.add(aspect)

    # ── Phase 2: Model branches fallback / supplement ─────────────────────
    branch_results = predict_aspect_sentiment(text, model, tokenizer)
    for aspect, sentiment in branch_results.items():
        if aspect not in detected_aspects and aspect != "Khác 🔄":
            results.append({
                "aspect": aspect,
                "clause": text,
                "sentiment": sentiment,
            })
            detected_aspects.add(aspect)

    # Nếu vẫn rỗng, lấy tất cả từ branches (kể cả Khác)
    if not results:
        for aspect, sentiment in branch_results.items():
            results.append({
                "aspect": aspect,
                "clause": text,
                "sentiment": sentiment,
            })

    return results
