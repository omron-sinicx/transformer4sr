import torch

class PositionalEncodings(torch.nn.Module):
    def __init__(self, seq_length, d_model, dropout):
        super().__init__()
        self.seq_length = seq_length
        self.d_model = d_model
        self.dropout = torch.nn.Dropout(p=dropout)
    
    def forward(self, x):
        pe = torch.zeros(self.seq_length, self.d_model)
        numerator = torch.arange(0, self.seq_length).unsqueeze(1)
        denominator = torch.pow(10e4, torch.arange(0, self.d_model, 2) / self.d_model).unsqueeze(0)
        pe[:, 0::2] = torch.sin(numerator / denominator)
        pe[:, 1::2] = torch.cos(numerator / denominator)
        pe.requires_grad_(False)
        return self.dropout(x + pe)