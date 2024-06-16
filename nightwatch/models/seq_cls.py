"""Sequence classifier models for sleep phase prediction."""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class LSTMSeqClassifier(nn.Module):
    def __init__(self, num_features, num_units, num_labels, num_layers=1, dropout=0.2):
        """LSTM-based sequence classifier model with batch normalization.

        Args:
        num_features (int): Number of input features.
        num_units (int): Number of hidden units in the LSTM.
        num_labels (int): Number of output classes.
        num_layers (int, optional): Number of LSTM layers. Defaults to 1.
        dropout (float, optional): Dropout rate. Defaults to 0.2.
        """
        super(LSTMSeqClassifier, self).__init__()
        
        self.norm = nn.BatchNorm1d(num_features)
        self.lstm = nn.LSTM(num_features, num_units,
                            num_layers, batch_first=True,
                            dropout=dropout)
        self.fc = nn.Linear(num_units, num_labels)

    def forward(self, x, seq_len):
        # Apply normalization
        B, T, F = x.shape
        x = x.view(B * T, F)
        x = self.norm(x)
        x = x.view(B, T, F)

        packed_input = pack_padded_sequence(x, seq_len.cpu(),
                                            batch_first=True,
                                            enforce_sorted=False)
        _, (ht, ct) = self.lstm(packed_input)

        out = self.fc(ht[-1])
        return out
