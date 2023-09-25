import torch
from model.mlp import MLP
from model.encoder_layer_mlp import EncoderLayerMLP
from model.encoder_layer_att import EncoderLayerAtt
from model.encoder_layer_mix import EncoderLayerMix

class Encoder(torch.nn.Module):
    def __init__(self, enc_type, nb_samples, max_nb_var, d_model, h, N, dropout):
        super().__init__()
        self.nb_samples = nb_samples
        self.max_nb_var = max_nb_var
        self.first_mlp = MLP([1, d_model, d_model], dropout)
        if enc_type=='mlp':
            self.layers = torch.nn.ModuleList([EncoderLayerMLP(nb_samples, max_nb_var, d_model, dropout) for _ in range(N)])
        elif enc_type=='att':
            self.layers = torch.nn.ModuleList([EncoderLayerAtt(nb_samples, max_nb_var, d_model, h, dropout) for _ in range(N)])
        elif enc_type=='mix':
            self.layers = torch.nn.ModuleList([EncoderLayerMix(nb_samples, max_nb_var, d_model, h, dropout) for _ in range(N)])
        else:
            raise ValueError('Encoder type should be \'mlp\', \'att\' or \'mix\'.')
        self.last_mlp = MLP([d_model, d_model], dropout)
    
    def forward(self, x):
        output = self.first_mlp(x)
        for layer in self.layers:
            output = layer(output)
        output = self.last_mlp(output)
        output = torch.max(output, dim=1)[0]  # make it permutation-invariant w.r.t. sample points
        return output