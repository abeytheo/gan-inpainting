import torch
from torch import nn

"""
    To avoid infinity due to derivative of sqrt(y) at y = 0 , 
    add some small epsilon
"""
class RMSELoss(nn.Module):
    def __init__(self, eps=1e-16):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss