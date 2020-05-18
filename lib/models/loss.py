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

"""
    Calculate loss on affected pixels denoted by binary masks `m`
"""
class LocalLoss(nn.Module):
    def __init__(self, baseloss, eps=1e-16):
        super().__init__()
        
        self.loss = baseloss(reduction='none')
        
        if isinstance(self.loss, RMSELoss):
            self.loss = nn.MSELoss(reduction='none')
            
        self.eps = eps
        
    def forward(self,yhat,y,mask):
        
        ### calculate loss on masked region
        loss = self.loss(y*mask,yhat*mask).sum()
        affected_pixels = (mask != 0).float().sum()
        
        ### micro-averaging on affected pixels
        loss = loss / affected_pixels
        
        if isinstance(self.loss, RMSELoss):
            loss = torch.sqrt(loss + self.eps)
            
        return loss