"""Sequence classifier models for sleep phase prediction."""

import torch
from torch import nn
from torch.nn import functional as f
from torch.nn.utils.rnn import pack_padded_sequence


class LSTMSeqClassifier(nn.Module):
    def __init__(self, num_features, num_units,
                 num_labels, num_layers=1, dropout=0.2):
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
        
        self.fc1 = nn.Linear(num_units, num_units // 2)
        self.fc2 = nn.Linear(num_units // 2, num_labels)


    def forward(self, x):
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        
        out, (ht, ct) = self.lstm(x)
        x = f.gelu(self.fc1(out[:, -1]))
        x = self.fc2(x)


        return x
