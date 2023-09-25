import torch
from model.mlp import MLP
from model.multihead_attention import MultiHeadAttention
from model.add_and_norm import AddAndNorm

class EncoderLayerMix(torch.nn.Module):
    def __init__(self, nb_samples, max_nb_var, d_model, h, dropout):
        super().__init__()
        self.mlp = MLP([d_model*max_nb_var, d_model, d_model], dropout)
        self.multihead_attention = MultiHeadAttention(h, d_model)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.add_norm = AddAndNorm(d_model)
    
    def forward(self, x):
        x_flat = torch.flatten(x, start_dim=2)  # (bs, nb_samples, max_nb_var*d_model)
        mlp_output = self.mlp(x_flat)  # (bs, nb_samples, d_model)
        mha_output = self.multihead_attention(mlp_output, mlp_output, mlp_output, mask=None)
        mha_output = self.dropout(mha_output)
        mha_output = torch.unsqueeze(mha_output, dim=2)
        addnorm_output = self.add_norm(x, mha_output)
        return addnorm_output