import torch
import torch.nn as nn

class GRUEncoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 bidirectional: bool = True,
                 dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(input_dim,
                          hidden_dim,
                          num_layers=num_layers,
                          batch_first=True,
                          bidirectional=bidirectional,
                          dropout=dropout)
        self.output_dim = hidden_dim * (2 if bidirectional else 1)

    def forward(self, x):
        # x: (B, T, D)
        out, h_n = self.gru(x)
        # h_n: (num_layers * num_directions, B, hidden_dim)
        if self.gru.bidirectional:
            h = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            h = h_n[-1]
        return h  # (B, output_dim)

