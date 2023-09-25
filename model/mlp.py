import torch

class MLP(torch.nn.Module):
    def __init__(self, list_dims, dropout):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for i in range(len(list_dims)-1):
            self.layers.append(torch.nn.Linear(list_dims[i], list_dims[i+1]))
            self.layers.append(torch.nn.ReLU())
            self.layers.append(torch.nn.Dropout(p=dropout))
    
    def forward(self, x):
        output = x
        for layer in self.layers:
            output = layer(output)
        return output