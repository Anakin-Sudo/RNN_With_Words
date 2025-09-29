# Deep RNNs for NLP: Sentiment Analysis and Text Generation

This project explores the use of **Recurrent Neural Networks (RNNs)** for two natural language processing (NLP) tasks:

1. **Movie Review Sentiment Analysis** with LSTM and BiLSTM architectures  
   → Models trained to classify reviews as positive or negative, deployed using **FastAPI** and **Streamlit**, and containerized with **Docker**.  

2. **Text Generation with RNNs** on Shakespeare’s works  
   → An experiment in next-word generation with a vanilla RNN. Results illustrate the **limitations of simple RNNs** on small datasets, motivating the use of more advanced architectures like LSTMs and Transformers but most 
importantly the weight of data size when it comes to training for general language understanding or generation.

Together, these two tasks showcase both the **strengths** and **limits** of RNN-based models for NLP.

---

## 📂 Project Structure

```
RNN_LSTM_Text_Generation/
│
├── utils/                          # Utility modules
│   ├── models.py                   # LSTM & BiLSTM model classes (with fixed hyperparams)
│   └── preprocessing.py             # Tokenizer, vocab handling, encoding, inference
│
├── best_lstm.pt                    # Trained LSTM weights
├── best_bilstm.pt                  # Trained BiLSTM weights
├── best_word_rnn.pt                # Trained vanilla RNN weights (Shakespeare)
│
├── vocab.json                      # Vocabulary (string→index)
├── itos.json                       # Index→string mapping
│
├── Movie_Review_Sentiment_Analysis.ipynb   # Training & evaluation of LSTM & BiLSTM
├── RNN_Shakespeare_Word_Prediction.ipynb   # RNN next-word generation experiment
│
├── review_mood_detector.py         # FastAPI backend (serves sentiment models)
├── review_mood_detector_front.py   # Streamlit frontend
│
├── requirements.txt                # Base training dependencies
├── requirements_deployment.txt     # Deployment dependencies
├── Dockerfile                      # Container definition
```

---

## Features

- **Sentiment Analysis (LSTM & BiLSTM)**
  - Preprocessing with tokenization, padding, and `<unk>` handling
  - PyTorch implementations of LSTM and BiLSTM
  - Deployment-ready: FastAPI endpoints + Streamlit UI + Docker

- **RNN Text Generation (Shakespeare)**
  - Vanilla RNN for next-word prediction
  - Highlights exposure bias and dataset size limitations
  - Commentary in the notebook on why results are poor

---

## Setup & Usage

### 1. Clone the repo
```bash
git clone https://github.com/your-username/RNN_LSTM_Text_Generation.git
cd RNN_LSTM_Text_Generation
```

### 2. Install dependencies
For training:
```bash
pip install -r requirements.txt
```

For deployment:
```bash
pip install -r requirements_deployment.txt
```

### 3. Run the API
```bash
uvicorn review_mood_detector:app --reload --host 0.0.0.0 --port 8000
```

### 4. Run the frontend
```bash
streamlit run review_mood_detector_front.py
```

### 5. Pre-Built Container

A pre-built Docker image is available on **GitHub Container Registry (GHCR)**:  

```bash
docker pull ghcr.io/anakin-sudo/sentiment_app:latest
docker run -it -p 8000:8000 -p 8501:8501 ghcr.io/anakin-sudo/sentiment_app:latest
```
---


## Next Steps

- Extend sentiment analysis to larger datasets (e.g., IMDB full dataset)  
- Replace vanilla RNN in text generation with LSTM or Transformer-based models  
- Add CI/CD workflows for automated serving  

---
