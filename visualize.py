import warnings
import pandas as pd
import re, os, math
import numpy as np

import torch
from torchvision import datasets, transforms, models

from lib.data.dataset import InpaintingDataset
from lib.models import networks

import matplotlib.pyplot as plt
import skimage
from skimage.color import grey2rgb

image_target_size = (128,128)
constant_ep = 50

device = torch.device('cpu')
if torch.cuda.is_available():
  device = torch.device('cuda:0')

def plot_inpainting_results(m_paths,csv_path):
  ### construct test dataset
  warnings.filterwarnings('ignore')
  test_df = pd.read_csv(csv_path)
  test_dataset = InpaintingDataset('/home/s2125048/thesis/dataset/',dataframe=test_df,
                                    transform=transforms.Compose([
                            transforms.Resize(image_target_size,),
                            transforms.ToTensor()]))

  test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=100,
    num_workers=0,
    shuffle=False
  )

  batch_size = 8
  num_batches = math.ceil(len(m_paths)/batch_size)

  for n in range(num_batches):
    print('batch',n)
    curr_m_paths = m_paths[n*batch_size:(n+1)*batch_size]

    net_G = {}
    for idx,path in enumerate(curr_m_paths):
      if torch.cuda.is_available():
        G_statedict = torch.load(os.path.join(path))
      else:
        G_statedict = torch.load(os.path.join(path),map_location='cpu')

      net_G[idx] = networks.get_network('generator','unet').to(device)
      net_G[idx].load_state_dict(G_statedict)
    
    start_ep = re.search('epoch\d+', curr_m_paths[0]).group(0)
    end_ep = re.search('epoch\d+', curr_m_paths[-1]).group(0)

    bcount = 0
    for test_loader_idx,(ground,mask,_) in enumerate(test_loader):  

      if not test_loader_idx % 3 == 0:
        continue 
        
      if bcount > 10:
        break

      print(test_loader_idx)
      ground = ground.to(device)
      mask = torch.ceil(mask).to(device)
      masked = ground * (1-mask)
      out = {}

      for idx,path in enumerate(curr_m_paths):
        out[idx] = net_G[idx](masked)
        out[idx] = masked + mask*out[idx]

      list_image_index = (np.array(range(0,10))*10).tolist()
      cols = 2 + batch_size
      plt.figure(figsize=(10,len(list_image_index)*3))

      for i, im_idx in enumerate(list_image_index):

        plt.subplot(len(list_image_index),cols,(i*cols+1))
        plt.imshow(ground[im_idx][0].detach().cpu().numpy(),cmap='Greys_r')
        plt.axis('off')
        if i == 0:
          plt.title('input')

        plt.subplot(len(list_image_index),cols,(i*cols+2))
        plt.imshow(masked[im_idx][0].detach().cpu().numpy(),cmap='Greys_r')
        plt.axis('off')
        if i == 0:
          plt.title('masked')

        for k in range(batch_size):
          plt.subplot(len(list_image_index),cols,(i*cols+3+k))
          im = (out[k][im_idx][0].detach().cpu().numpy()*255).astype(np.uint8)
          im = grey2rgb(im)
          plt.imshow(im)
          plt.axis('off')
          if i == 0:
            plt.title('ep {}'.format(constant_ep * (n*batch_size + k + 1)))

      plt.savefig(f'{result_root}/{start_ep}-{end_ep}_batch{test_loader_idx}.png',dpi=300)
      bcount += 1
### plotting

import re
from operator import itemgetter

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

print(new_paths)

### dataset

dataset = '/home/s2125048/thesis/dataset/csv/test_all_masks.csv'
result_root = f'{exp_root}/results/test/'

if not os.path.exists(result_root):
  os.makedirs(result_root)

plot_inpainting_results(new_paths,dataset)

dataset = '/home/s2125048/thesis/dataset/csv/extra.csv'
result_root = f'{exp_root}/results/extra/'

if not os.path.exists(result_root):
  os.makedirs(result_root)

plot_inpainting_results(new_paths,dataset)