
import warnings
import pandas as pd
import re, os, math
import numpy as np

import pickle
import torch
from torch import nn
from torchvision import datasets, transforms, models

from lib.data.dataset import InpaintingDataset
from lib.models import networks

import matplotlib.pyplot as plt
import skimage
from skimage.color import grey2rgb

import re
from operator import itemgetter

class LocalLoss(nn.Module):
    def __init__(self, baseloss):
        super().__init__()
        
        self.loss = baseloss(reduction='none')
            
        self.eps = eps
        
    def forward(self,yhat,y,mask):
        
        ### calculate loss on masked region
        loss = self.loss(y*mask,yhat*mask).sum()
        affected_pixels = (mask != 0).float().sum()
        
        ### micro-averaging on affected pixels
        loss = loss / affected_pixels
            
        return loss

exp = 'wgan_rmse'
exp_root = f'/home/s2125048/thesis/model/{exp}/'

### ep paths

model_paths = []
for root,dirs,files in os.walk(exp_root):
  for f in files:
    if '.pt' in f:
      d = {}
      d['path'] = os.path.join(root,f)
      d['ep'] = int(re.search('epoch\d+', f).group(0).split('epoch')[1])
      model_paths.append(d)

new =  sorted(model_paths, key=itemgetter('ep'))
new_paths = [p['path'] for p in new]

image_target_size = 128
dataset_path = "/home/s2125048/thesis/dataset/"

### construct training dataset
train_df = pd.read_csv(os.path.join(dataset_path,'csv/train_all_masks.csv'))

train_dataset = dataset.InpaintingDataset(dataset_path,dataframe=train_df,
                                  transform=transforms.Compose([
                          transforms.Resize(image_target_size,),
                          transforms.ToTensor()]))

train_loader = torch.utils.data.DataLoader(
  train_dataset,
  batch_size=64,
  num_workers=0,
  shuffle=True
)

### construct test dataset
test_df = pd.read_csv(os.path.join(dataset_path,'csv/test_all_masks.csv'))

test_dataset = dataset.InpaintingDataset(dataset_path,dataframe=test_df,
                                  transform=transforms.Compose([
                          transforms.Resize(image_target_size,),
                          transforms.ToTensor()]))
test_loader = torch.utils.data.DataLoader(
  test_dataset,
  batch_size=64,
  num_workers=0,
  shuffle=False
)

metric = {
  'train': {},
  'test': {}
}
mse_local_criterion = LocalLoss(nn.MSELoss)

for p in new_paths:

  G_statedict = torch.load(p,map_location='cpu')
  net_G = get_network('generator','unet').cpu()
  net_G.load_state_dict(G_statedict)

  ep = int(re.search('epoch\d+', curr_m_paths[0]).group(0).split('epoch')[-1])
  print('epoch',p)

  train_loss = 0
  test_loss = 0
  
  print("train")
  b = 0
  for ground,mask,_ in train_loader:
    b+=1
    mask = torch.ceil(mask)
    masked = ground * (1-mask)
    out = net_G(masked)
    train_loss += mse_local_criterion(out,ground,mask)
  train_loss = train_loss / b

  print("test")
  b = 0
  for ground,mask,_ in test_loader:
    b+=1
    mask = torch.ceil(mask)
    masked = ground * (1-mask)
    out = net_G(masked)
    test_loss += mse_local_criterion(out,ground,mask)
  test_loss = test_loss / b

  metric['train'][ep] = train_loss
  metric['test'][ep] = test_loss
  
  with open(os.path.join(exp_root,f'local_mse_{exp}.obj'),'wb') as handle:
    pickle.dump(metric, handle, protocol=pickle.HIGHEST_PROTOCOL)
      





