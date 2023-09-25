import torch
from model.multihead_attention import MultiHeadAttention
from model.add_and_norm import AddAndNorm

class EncoderLayerAtt(torch.nn.Module):
    def __init__(self, nb_samples, max_nb_var, d_model, h, dropout):
        super().__init__()
        self.multihead_attention = MultiHeadAttention(h, d_model)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.add_norm = AddAndNorm(d_model)
    
    def forward(self, x):
        mha_output = self.multihead_attention(x, x, x, mask=None)
        mha_output = self.dropout(mha_output)
        addnorm_output = self.add_norm(x, mha_output)
        return addnorm_output