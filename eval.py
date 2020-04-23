import torch
import pandas as pd
import time
import pickle
from torchvision import datasets, transforms
from torch import optim, nn
import os, logging, argparse
import re
import pickle

from lib.models import networks, util, evaluate
from lib.data import dataset
import lib.pytorch_ssim as pytorch_ssim

from datetime import datetime as dt

ts = dt.strftime(dt.now(),'%Y%m%d_%H%M%S')

print(ts)

### Parse arguments
parser = argparse.ArgumentParser(description='Training configurations')

parser.add_argument('-m','--modelpath', nargs='?', type=str, required=True, help='Model directory')
parser.add_argument('-d','--datasetpath', nargs='?', type=str, required=True, help='Dataset directory')
parser.add_argument('-f','--csvfile', nargs='?', type=str, required=True, help='CSV file for dataset dataframe')
parser.add_argument('-g','--generator', nargs='?', type=str, choices=['unet','vgg19'], default='unet', help='Generator network name')

args = parser.parse_args()

device=None
if torch.cuda.is_available():
  device = torch.device("gpu:0")
else:
  device = torch.device("cpu")

### construct eval dataset
eval_df = pd.read_csv(args.csvfile)
def replace_path(x,ds_path):
  x['mask_source'] = os.path.join(ds_path,x['mask_source'])
  x['groundtruth_source'] = os.path.join(ds_path,x['groundtruth_source'])
  return x
eval_df = eval_df.apply(lambda x: replace_path(x,args.datasetpath),1)
eval_dataset = dataset.InpaintingDataset('',dataframe=eval_df,
                                  transform=transforms.Compose([
                          transforms.ToTensor()]))

eval_loader = torch.utils.data.DataLoader(
  eval_dataset,
  batch_size=50,
  num_workers=0,
  shuffle=False
)

# ev = next(iter(eval_loader))
# ground = ev[0]
# mask = ev[1]

# if 'real' in args.csvfile:
#   masked = ground * mask
# else:
#   masked = ground * (1-mask)

### evaluate
model_path = args.modelpath
metrics = {}
out = {}
ground = None
mask = None
masked = None

for m in os.listdir(model_path):
    if '.pt' in m:
        epoch = int(re.search('epoch(\d+?)\_', m).group(1))
        full_path = os.path.join(model_path,m)
        print('eval',m)
        net_G = networks.get_network('generator',args.generator)
        G_statedict = torch.load(full_path,map_location=device)
        net_G.load_state_dict(G_statedict)
        
        # batch_ssim = torch.tensor([],dtype=torch.float32)
        # batch_l2 = torch.tensor([],dtype=torch.float32)
        # batch_ssim_local = torch.tensor([],dtype=torch.float32)
        # batch_l2_local = torch.tensor([],dtype=torch.float32)

        for gt,m,_ in eval_loader:

          ground = gt
          mask = m

          if 'real' in args.csvfile or 'extra' in args.csvfile:
            masked = ground * mask
          else:
            masked = ground * (1-mask)
          
          out[epoch] = net_G(masked.cpu())
          # ssim = pytorch_ssim.ssim(ground, out[epoch])
          # l2 = torch.dist(ground,out[epoch])

          # if 'real' in args.csvfile:
          #   ssim_local = pytorch_ssim.ssim(ground * (1-mask), out[epoch]*(1-mask))
          #   l2_local = torch.dist(ground * (1-mask), out[epoch]*(1-mask))
          # else:
          #   ssim_local = pytorch_ssim.ssim(ground * mask, out[epoch]*mask)
          #   l2_local = torch.dist(ground * mask, out[epoch]*mask)
          
          # batch_ssim = torch.cat((batch_ssim,ssim))
          # batch_l2 = torch.cat((batch_l2,l2))
          # batch_ssim_local = torch.cat((batch_ssim_local,ssim_local))
          # batch_l2_local = torch.cat((batch_l2_local,l2_local))
        
        # metrics[epoch] = {
        #   'ssim': batch_ssim.mean().item(),
        #   'l2': batch_l2.mean().item(),
        #   'ssim_local': batch_ssim_local.mean().item(),
        #   'l2_local': batch_l2_local.mean().item()
        # }

with open(os.path.join(model_path,'{}_input.obj'.format(ts)),'wb') as handle:
  inp = {
    'ground': ground.cpu().numpy(),
    'mask': mask.cpu().numpy(),
    'masked': masked.cpu().numpy()
  }
  pickle.dump(inp, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(os.path.join(model_path,'{}_output.obj'.format(ts)),'wb') as handle:
  pickle.dump(out, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open(os.path.join(model_path,'{}_metric.obj'.format(ts)),'wb') as handle:
#   pickle.dump(metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)