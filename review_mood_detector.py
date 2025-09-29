import torch
from fastapi import FastAPI
from pydantic import BaseModel
from utils.preprocessing import predict_sentiment
from utils.models import LSTMClassifier, BiLSTMClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Request schema
class ReviewRequest(BaseModel):
    text: str


app = FastAPI(title="Sentiment Analysis API (LSTM & BiLSTM)")

# Load models (no hyperparams needed)
lstm_model = LSTMClassifier().to(device)
lstm_model.load_state_dict(torch.load("best_lstm.pt", map_location=device))
lstm_model.eval()

bilstm_model = BiLSTMClassifier().to(device)
bilstm_model.load_state_dict(torch.load("best_bilstm.pt", map_location=device))
bilstm_model.eval()


# Endpoints
@app.post("/predict_lstm")
def predict_lstm(request: ReviewRequest):
    pred_class, probs = predict_sentiment(lstm_model, request.text, device)
    return {"model": "LSTM", "prediction": int(pred_class), "probs": probs}


@app.post("/predict_bilstm")
def predict_bilstm(request: ReviewRequest):
    pred_class, probs = predict_sentiment(bilstm_model, request.text, device)
    return {"model": "BiLSTM", "prediction": int(pred_class), "probs": probs}
