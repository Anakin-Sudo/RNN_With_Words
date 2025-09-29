import re
import torch
import json

# Load vocab
with open("vocab.json") as f:
    stoi = json.load(f)


# Tokenizer
def simple_tokenizer(text: str):
    return re.findall(r"\b\w+\b", text.lower())


# Encode
def encode(text: str, max_len: int = 200) -> torch.Tensor:
    tokens = simple_tokenizer(text)
    ids = [stoi.get(tok, 1) for tok in tokens]  # 1 = <unk>
    return torch.tensor(ids[:max_len], dtype=torch.long)


# Prediction helper
def predict_sentiment(model, text: str, device, max_len: int = 200):
    model.eval()
    tokens = encode(text, max_len=max_len)
    length = torch.tensor([len(tokens)], dtype=torch.long)

    tokens = tokens.unsqueeze(0).to(device)  # shape (1, seq_len)
    length = length.to(device)

    with torch.no_grad():
        outputs = model(tokens, length)
        probs = torch.softmax(outputs, dim=1)
        pred_class = probs.argmax(1).item()

    return pred_class, probs.squeeze().tolist()
