import torch

class AddAndNorm(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm(d_model)
    
    def forward(self, x_input, x_output):
        output = self.layer_norm(x_input + x_output)  # residual connection
        return output