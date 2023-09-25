import math
import torch

class TokenEmbeddings(torch.nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
    
    def forward(self, x):
        return self.embed(x) * math.sqrt(self.d_model)