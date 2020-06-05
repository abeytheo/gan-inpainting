import torch
from torch import nn
from lib.models import networks

device = torch.device('cpu')
if torch.cuda.is_available():
  device = torch.device("cuda:0")
  
vgg = networks.VGG19Wrapper().to(device)
vgg.eval()

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
        if isinstance(baseloss, RMSELoss):
            self.loss = nn.MSELoss(reduction='none')
        else:
          self.loss = baseloss(reduction='none')       
            
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

def perceptual_loss(output,target,weight=0.05):

  with torch.no_grad():

    ### relu_1_1, relu_2_1, relu_3_1, relu_4_1, relu_5_1 of VGG19
    feature_maps_indices = [1,6,11,20,29]
    prev_o = output.repeat(1,3,1,1).detach()
    prev_t = target.repeat(1,3,1,1).detach()

    p_loss = 0

    for i,m in enumerate(vgg.vgg19.features._modules.values()):
      
      next_o = m(prev_o)
      next_t = m(prev_t)
      if i in feature_maps_indices:
        p_loss += torch.mean(torch.pow(next_o - next_t,2))
      prev_o = next_o
      prev_t = next_t
    
    return weight*p_loss

def style_loss(output,target,weight=0.1):

  with torch.no_grad():

    ### relu_1_1, relu_2_1, relu_3_1, relu_4_1, relu_5_1 of VGG19
    feature_maps_indices = [1,6,11,20,29]
    prev_o = output.repeat(1,3,1,1).detach()
    prev_t = target.repeat(1,3,1,1).detach()

    s_loss = 0

    for i,m in enumerate(vgg.vgg19.features._modules.values()):
      
      next_o = m(prev_o)
      next_t = m(prev_t)
      if i in feature_maps_indices:
        s_loss += torch.mean(torch.pow(gram_matrix(next_o) - gram_matrix(next_t),2))
      prev_o = next_o
      prev_t = next_t
    
    return weight*s_loss

def perceptual_and_style_loss(output,target,weight_p=0.05,weight_s=100):

  with torch.no_grad():

    ### relu_1_1, relu_2_1, relu_3_1, relu_4_1, relu_5_1 of VGG19
    feature_maps_indices = [1,6,11,20,29]
    prev_o = output.repeat(1,3,1,1).detach()
    prev_t = target.repeat(1,3,1,1).detach()

    p_loss = 0
    s_loss = 0

    for i,m in enumerate(vgg.vgg19.features._modules.values()):
      
      next_o = m(prev_o)
      next_t = m(prev_t)
      if i in feature_maps_indices:
        p_loss += torch.mean(torch.pow(next_o - next_t,2))
        s_loss += torch.mean(torch.pow(gram_matrix(next_o) - gram_matrix(next_t),2))
      prev_o = next_o
      prev_t = next_t
    
    return weight_p*p_loss, weight_s*s_loss

def gram_matrix(features, normalize=True):
    """
    Compute the Gram matrix from features.
    Inputs:
    - features: PyTorch Variable of shape (N, C, H, W) giving features for
      a batch of N images.
    - normalize: optional, whether to normalize the Gram matrix
        If True, divide the Gram matrix by the number of neurons (H * W * C)
    Returns:
    - gram: PyTorch Variable of shape (N, C, C) giving the
      (optionally normalized) Gram matrices for the N input images.
    """
    N, C, H, W = features.size()
    # Use torch.bmm for batch multiplication of matrices
    feat_reshaped = features.view(N, C, -1)
    gram = torch.bmm(feat_reshaped, feat_reshaped.transpose(1, 2))
    if normalize:
        return gram / (H*W*C)
    else:
        return gram

def tv_loss(img, tv_weight):
    """
    Compute total variation loss.
    Inputs:
    - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.
    Returns:
    - loss: PyTorch Variable holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """
    w_variance = torch.mean(torch.pow(img[:,:,:,:-1] - img[:,:,:,1:], 2))
    h_variance = torch.mean(torch.pow(img[:,:,:-1,:] - img[:,:,1:,:], 2))
    loss = tv_weight * (h_variance + w_variance)
    return loss