import streamlit as st
import requests

st.title("🎬 Movie Review Sentiment Analysis: LSTM vs BiLSTM")

api_url = "http://localhost:8000"

review = st.text_area("Enter a movie review:", "The movie was awfully boring, cannot recommend it.")

col1, col2 = st.columns(2)

if col1.button("Analyze with LSTM"):
    if review.strip():
        lstm_resp = requests.post(f"{api_url}/predict_lstm", json={"text": review})
        if lstm_resp.status_code == 200:
            lstm_result = lstm_resp.json()
            st.subheader("LSTM Result")
            st.write("Prediction:", "Positive" if lstm_result["prediction"] == 1 else "Negative")
            st.write("Probabilities:", lstm_result["probs"])
        else:
            st.error("Error: could not reach API")

if col2.button("Analyze with BiLSTM"):
    if review.strip():
        bilstm_resp = requests.post(f"{api_url}/predict_bilstm", json={"text": review})
        if bilstm_resp.status_code == 200:
            bilstm_result = bilstm_resp.json()
            st.subheader("BiLSTM Result")
            st.write("Prediction:", "Positive ❤️" if bilstm_result["prediction"] == 1 else "Negative ❌")
            st.write("Probabilities:", bilstm_result["probs"])
        else:
            st.error("Error: could not reach API")
