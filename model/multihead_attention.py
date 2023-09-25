import torch
import math

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, h, d_model):
        super().__init__()
        self.h = h
        self.d_model = d_model
        assert (d_model % h == 0)
        self.d_k = d_model // h  # we assume d_k = d_v = d_model / h
        self.W_Q = torch.nn.Linear(d_model, d_model)  # learned projections for queries, ...
        self.W_K = torch.nn.Linear(d_model, d_model)  # keys, ...
        self.W_V = torch.nn.Linear(d_model, d_model)  # and values
        self.W_O = torch.nn.Linear(d_model, d_model)  # this is for the multi-head attention output
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = torch.where(mask, torch.Tensor([-1e9]), scores)
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, V)
    
    def forward(self, Q, K, V, mask=None):
        Q_temp = torch.reshape(self.W_Q(Q), ([i for i in Q.shape[:-1]]+[self.h]+[self.d_k])).transpose(1, 2)
        K_temp = torch.reshape(self.W_K(K), ([i for i in K.shape[:-1]]+[self.h]+[self.d_k])).transpose(1, 2)
        V_temp = torch.reshape(self.W_V(V), ([i for i in V.shape[:-1]]+[self.h]+[self.d_k])).transpose(1, 2)
        sdpa = self.scaled_dot_product_attention(Q_temp, K_temp, V_temp, mask).transpose(1, 2)
        sdpa = torch.reshape(sdpa, ([i for i in sdpa.shape[:-2]]+[self.d_model]))
        mha_output = self.W_O(sdpa)
        return mha_output