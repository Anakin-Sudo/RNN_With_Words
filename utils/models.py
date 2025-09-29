import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import torch


# LSTM Classifier with fixed hyperparams
class LSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # Hardcoded params (must match training)
        self.vocab_size = 20000
        self.embed_dim = 200
        self.hidden_dim = 256
        self.num_classes = 2
        self.dropout_rate = 0.5

        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(self.embed_dim, self.hidden_dim, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, self.num_classes)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        out = self.dropout(h[-1])
        return self.fc(out)


# BiLSTM Classifier with fixed hyperparams
class BiLSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # Hardcoded params (matching training hyperparams)
        self.vocab_size = 20000
        self.embed_dim = 200
        self.hidden_dim = 256
        self.num_classes = 2
        self.dropout_rate = 0.5

        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(self.embed_dim, self.hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(self.hidden_dim * 2, self.num_classes)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        h_forward, h_backward = h[-2], h[-1]
        h_cat = torch.cat((h_forward, h_backward), dim=1)
        return self.fc(self.dropout(h_cat))
