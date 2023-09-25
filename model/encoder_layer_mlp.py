import torch
from model.mlp import MLP

class EncoderLayerMLP(torch.nn.Module):
    def __init__(self, nb_samples, max_nb_var, d_model, dropout):
        super().__init__()
        self.nb_samples = nb_samples
        self.max_nb_var = max_nb_var
        assert (d_model % 2 == 0)
        self.mlp1 = MLP([d_model, int(d_model/2), int(d_model/2)], dropout)
        self.mlp2 = MLP([d_model, int(d_model/2), int(d_model/2)], dropout)
    
    def forward(self, x):
        mlp1_output = self.mlp1(x)  # (bs, nb_samples, max_nb_var, d_model/2)
        var_feats = torch.max(mlp1_output, dim=1)[0]  # obtain variable-wise features
        var_feats = torch.tile(var_feats.unsqueeze(1), dims=[1, self.nb_samples, 1, 1])
        concat1 = torch.cat((mlp1_output, var_feats), dim=3)  # (bs, nb_samples, max_nb_var, d_model)
        mlp2_output = self.mlp2(concat1)  # (bs, nb_samples, max_nb_var, d_model/2)
        pts_feats = torch.max(mlp2_output, dim=2)[0]  # obtain point-wise features
        pts_feats = torch.tile(pts_feats.unsqueeze(2), dims=[1, 1, self.max_nb_var, 1])
        concat2 = torch.cat((mlp2_output, pts_feats), dim=3)  # (bs, nb_samples, max_nb_var, d_model)
        return concat2