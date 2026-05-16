"""
preprocessor.py
Tiền xử lý văn bản tiếng Việt cho TLU Feedback Analytics.
"""
import re
import unicodedata

# Từ nối dùng để tách mệnh đề
CLAUSE_SEPARATORS = [
    r'\bnhưng\b',
    r'\btuy nhiên\b',
    r'\bdù vậy\b',
    r'\bdù\b',
    r'\bmặc dù\b',
    r'\bthế nhưng\b',
    r'\bcòn\b',
    r'\bsong\b',
    r'\bchỉ là\b',
]

def normalize_text(text: str) -> str:
    """Chuẩn hóa Unicode về dạng NFC (đúng chuẩn tiếng Việt)."""
    return unicodedata.normalize('NFC', text.strip())

def clean_text(text: str) -> str:
    """
    Làm sạch văn bản: xóa ký tự thừa, chuẩn hóa khoảng trắng.
    Giữ lại dấu câu tiếng Việt và các emoji cơ bản.
    """
    text = normalize_text(text)
    # Xóa URL
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Xóa HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Giữ lại chữ, số, dấu câu cơ bản
    text = re.sub(r'[^\w\s\.,!?;:()\-àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ]', ' ', text)
    # Chuẩn hóa khoảng trắng
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def split_clauses(text: str) -> list[str]:
    """
    Tách câu thành các mệnh đề theo:
    - Từ nối: nhưng, tuy nhiên, còn, dù, song...
    - Dấu phẩy (chỉ khi mệnh đề > 3 từ)
    Trả về list các clause không rỗng, đã strip.
    """
    # Tách theo từ nối trước
    sep_pattern = '|'.join(CLAUSE_SEPARATORS)
    parts = re.split(sep_pattern, text, flags=re.IGNORECASE)

    clauses = []
    for part in parts:
        # Tách thêm theo dấu phẩy, nhưng chỉ giữ clause >= 3 từ
        sub_parts = [p.strip() for p in part.split(',') if len(p.strip().split()) >= 3]
        if sub_parts:
            clauses.extend(sub_parts)
        elif part.strip():
            clauses.append(part.strip())

    # Lọc rỗng và trả về
    return [c for c in clauses if c]
