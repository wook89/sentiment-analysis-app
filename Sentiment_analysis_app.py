import os
import platform
import pickle
import numpy as np
import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ─── 한글 폰트 설정 ───
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우 기본 한글 폰트
elif platform.system() == 'Darwin':  # macOS
    plt.rcParams['font.family'] = 'AppleGothic'
else:
    plt.rcParams['font.family'] = 'NanumGothic'  # 리눅스 (사전 설치 필요)

# 음수 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

# ─── 0. GPU 메모리 설정 ───
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# ─── 1. 모델 & 토크나이저 로드 ───
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('best_model_GameReview.h5')

@st.cache_resource
def load_tokenizer():
    with open('game_review_sentiment.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()
tokenizer = load_tokenizer()

# ─── 2. 전역 상수 및 예측 함수 ───
IDX2LABEL = {0: "부정", 1: "중립", 2: "긍정"}
max_len = 100  # 학습 시 사용한 padding 길이

def predict_sentiment(review: str,
                      tokenizer,
                      model,
                      max_len: int,
                      idx2label: dict = IDX2LABEL) -> tuple[str, float, np.ndarray]:
    # 1) 정수 인코딩
    seq = tokenizer.texts_to_sequences([review])
    # 2) 패딩
    seq_pad = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
    # 3) 예측
    probs = model.predict(seq_pad, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = idx2label[pred_idx]
    confidence = float(probs[pred_idx])
    return pred_label, confidence, probs

# ─── 3. Streamlit UI ───
st.title("🎯 감성분석 리뷰 예측기")
st.write("리뷰 문장을 입력하면 부정/중립/긍정을 예측합니다.")

user_input = st.text_area("✍️ 리뷰를 입력하세요:")

if st.button("예측하기") and user_input.strip():
    # 예측 수행
    label, conf, probs = predict_sentiment(user_input, tokenizer, model, max_len)

    # 결과 출력
    st.markdown(f"## 📌 예측 결과: **{label}** ({conf*100:.2f}%)")
    st.markdown("### 📊 감정별 확률 (%):")
    for i, label_text in IDX2LABEL.items():
        st.markdown(f"- **{label_text}**: {probs[i]*100:.2f}%")

    # 시각화
    fig, ax = plt.subplots()
    ax.bar(IDX2LABEL.values(), probs * 100)
    ax.set_ylabel("Confidence (%)")
    ax.set_ylim(0, 100)
    st.pyplot(fig)
