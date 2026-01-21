import torch
import torch.nn as nn


class ForwardModel(nn.Module):
    def __init__(self, input_features=10, hidden_size=200, dropout_rate=0.0, num_layers=6):
        super().__init__()
        self.rnn = nn.LSTM(input_size=input_features, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout_rate, bidirectional=True)
        self.out = nn.Linear(hidden_size*2, 1)

    def forward(self,x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out, _ = self.rnn(x)
        last_out = out[:, -1, :]
        return nn.functional.softplus(self.out(last_out))
