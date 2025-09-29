FROM python:3.10-slim

WORKDIR /app

# Install Python dependencies
COPY requirements_deployment.txt .
RUN pip install --no-cache-dir -r requirements_deployment.txt

# Copy project files exactly as in your repo
COPY utils ./utils
COPY review_mood_detector.py .
COPY review_mood_detector_front.py .
COPY vocab.json .
COPY best_lstm.pt .
COPY best_bilstm.pt .

# Expose FastAPI and Streamlit ports
EXPOSE 8000
EXPOSE 8501

# Run FastAPI and Streamlit apps
CMD uvicorn review_mood_detector:app --host 0.0.0.0 --port 8000 & \
    streamlit run review_mood_detector_front.py --server.port=8501 --server.address=0.0.0.0
