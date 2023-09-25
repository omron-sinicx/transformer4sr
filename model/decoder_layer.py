import torch
from model.multihead_attention import MultiHeadAttention
from model.add_and_norm import AddAndNorm
from model.mlp import MLP

class DecoderLayer(torch.nn.Module):
    def __init__(self, h, d_model, dropout):
        super().__init__()
        self.multihead_attention1 = MultiHeadAttention(h, d_model)
        self.dropout1 = torch.nn.Dropout(p=dropout)
        self.add_norm1 = AddAndNorm(d_model)
        self.multihead_attention2 = MultiHeadAttention(h, d_model)
        self.dropout2 = torch.nn.Dropout(p=dropout)
        self.add_norm2 = AddAndNorm(d_model)
        self.mlp = MLP([d_model, 2*d_model, d_model], dropout)  # dropout already inside MLP
        self.add_norm3 = AddAndNorm(d_model)
    
    def forward(self, input_dec, mask_dec, output_enc):
        mha_output1 = self.multihead_attention1(input_dec, input_dec, input_dec, mask_dec)
        mha_output1 = self.dropout1(mha_output1)
        addnorm_output1 = self.add_norm1(input_dec, mha_output1)
        mha_output2 = self.multihead_attention2(addnorm_output1, output_enc, output_enc, mask=None)
        mha_output2 = self.dropout2(mha_output2)
        addnorm_output2 = self.add_norm2(addnorm_output1, mha_output2)
        mlp_output = self.mlp(addnorm_output2)  # dropout already inside MLP
        addnorm_output3 = self.add_norm3(addnorm_output2, mlp_output)
        return addnorm_output3