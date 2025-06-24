# 🎯 감성분석 리뷰 예측기 (Sentiment Analysis Web App)

텍스트 리뷰를 입력하면 부정 / 중립 / 긍정 중 하나로 예측하는 Streamlit 기반 웹 애플리케이션입니다.  
딥러닝 모델(LSTM + BiLSTM)을 로드하여, 실시간 예측 결과를 확률과 함께 시각화합니다.

---

## 📌 기능

- 사용자 입력 리뷰 → 감성 예측 (부정 / 중립 / 긍정)
- 예측 결과 + 확률 바 차트 시각화
- 한글 텍스트 완벽 지원

---

## 🖥️ 실행 방법

### 1. 필수 패키지 설치
```bash
pip install -r requirements.txt
```

### 2. Streamlit 앱 실행
```bash
streamlit run Sentiment_analysis_app.py
```
🧠 사용된 모델
Architecture: Embedding → BiLSTM → Dropout → Dense(softmax)

Loss Function: Categorical CrossEntropy

Tokenizer: Keras Tokenizer (저장 및 불러오기)

모델: https://drive.google.com/file/d/1FqGmrZoQjO-gcSPZoKXs7XMgSX-8qs_U/view?usp=drive_link

Tokenizer: [game_review_sentiment.pkl](https://drive.google.com/file/d/1u94r_rzMJrRfxQf-42b7RvXGHknVlEYR/view?usp=drive_link)


## 📸 예시 화면

> 리뷰: `재밌다`

![예시 화면](assets/sample_ui.png)
