"""
model_loader.py
Load và cache CNN+BiLSTM model + tokenizer cho TLU Feedback Analytics.
"""
import os
import pickle
import streamlit as st
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, Conv1D, GlobalMaxPooling1D,
    concatenate, Dense, Dropout, Bidirectional, LSTM
)

WEIGHTS_PATH = "sentiment_cnn_weights.weights.h5"
TOKENIZER_PATH = "tokenizer.pickle"
MAX_VOCAB_SIZE = 10000
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 50


def build_model() -> Model:
    """Xây dựng lại kiến trúc CNN + BiLSTM (khớp với train.py)."""
    input_layer = Input(shape=(MAX_SEQUENCE_LENGTH,))
    embedding_layer = Embedding(input_dim=MAX_VOCAB_SIZE, output_dim=EMBEDDING_DIM)(input_layer)

    # Multi-kernel CNN
    conv_blocks = []
    for kernel_size in [3, 4, 5]:
        conv = Conv1D(filters=128, kernel_size=kernel_size, activation='relu')(embedding_layer)
        pool = GlobalMaxPooling1D()(conv)
        conv_blocks.append(pool)
    cnn_concat = concatenate(conv_blocks, axis=1)

    # BiLSTM
    bilstm = Bidirectional(LSTM(64, return_sequences=False))(embedding_layer)

    # Kết hợp CNN + BiLSTM
    concat_layer = concatenate([cnn_concat, bilstm], axis=1)

    # 4 nhánh output: Lecturer, Training, Facility, Others
    def build_branch(name):
        dense1 = Dense(64, activation='relu')(concat_layer)
        drop1 = Dropout(0.5)(dense1)
        return Dense(4, activation='softmax', name=f'{name}_output')(drop1)

    out_lecturer = build_branch('lecturer')
    out_training = build_branch('training')
    out_facility = build_branch('facility')
    out_others = build_branch('others')

    return Model(inputs=input_layer, outputs=[out_lecturer, out_training, out_facility, out_others])


@st.cache_resource(show_spinner="🤖 Đang tải mô hình AI...")
def load_resources(weights_mtime: float = 0, tok_mtime: float = 0):
    """
    Load model weights và tokenizer, cache để không phải load lại mỗi lần refresh.
    weights_mtime và tok_mtime dùng để invalidate cache khi file thay đổi.
    """
    if not os.path.exists(WEIGHTS_PATH) or not os.path.exists(TOKENIZER_PATH):
        return None, None

    model = build_model()
    model.load_weights(WEIGHTS_PATH)

    with open(TOKENIZER_PATH, 'rb') as handle:
        tokenizer = pickle.load(handle)

    return model, tokenizer


def get_model_and_tokenizer():
    """Helper function: lấy model và tokenizer với cache-busting tự động."""
    weights_mtime = os.path.getmtime(WEIGHTS_PATH) if os.path.exists(WEIGHTS_PATH) else 0
    tok_mtime = os.path.getmtime(TOKENIZER_PATH) if os.path.exists(TOKENIZER_PATH) else 0
    return load_resources(weights_mtime, tok_mtime)
