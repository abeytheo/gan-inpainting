import importlib, os
import argparse
import pandas as pd
import numpy as np
import skimage
import skimage.io as io
from skimage.color import grey2rgb
import functools

import torch
from torch import nn
from torchvision import datasets, transforms

from lib.data import dataset
from lib.models import networks

from lib.fid.inception import InceptionV3
import lib.fid.fid_score as fid

### Training arguments

parser = argparse.ArgumentParser(description='Training configurations')

parser.add_argument('-exp','--experiments',nargs='+',type=str,required=True)
parser.add_argument('-ep','--numepoch', nargs='?', type=int, default=1500, help='Number of training epoch')
parser.add_argument('-b','--batchsize', nargs='?', type=int, default=128, help='Batch size in one training epoch')
parser.add_argument('-g','--generator', nargs='?', type=str, choices=['unet','vgg19'], default='unet', help='Generator network name')
parser.add_argument('-d','--discriminator', nargs='?', type=str, choices=['patchgan','dcgan'], default='patchgan',help='Discriminator network name')
parser.add_argument('--imagedim', nargs='?', type=int, default=128,help='Image dimension')
parser.add_argument('--saveevery', nargs='?', type=int, default=50,help='Save network every N epoch(s)')
parser.add_argument('--updatediscevery', nargs='?', type=int, default=3,help='Backprop discriminator every N epoch(s)')
parser.add_argument('--evalevery', nargs='?', type=int, default=10,help='Evaluate test set every')
parser.add_argument('--debug', nargs='?', type=str, default="false",help='Evaluate test set every')

args = parser.parse_args()
state = vars(args)

### Check experiment file
exp_list = os.listdir('experiment_list')
for ex in args.experiments:
  if ex+'.py' not in exp_list:
    raise Exception("Invalid experiments")

device = torch.device('cpu')
if torch.cuda.is_available():
  device = torch.device("cuda:0")

"""
  Convert string argument to boolean
"""
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

debug = str2bool(args.debug)
shuffle = True
image_target_size = args.imagedim
dataset_path = "/home/s2125048/thesis/dataset/"

### construct training dataset
train_df = pd.read_csv(os.path.join(dataset_path,'csv/train_all_masks.csv'))

train_dataset = dataset.InpaintingDataset(dataset_path,dataframe=train_df,
                                  transform=transforms.Compose([
                          transforms.Resize(image_target_size,),
                          transforms.ToTensor()]))

train_loader = torch.utils.data.DataLoader(
  train_dataset,
  batch_size=args.batchsize,
  num_workers=0,
  shuffle=shuffle
)

### construct test dataset
test_df = pd.read_csv(os.path.join(dataset_path,'csv/test_all_masks.csv'))

test_dataset = dataset.InpaintingDataset(dataset_path,dataframe=test_df,
                                  transform=transforms.Compose([
                          transforms.Resize(image_target_size,),
                          transforms.ToTensor()]))
test_loader = torch.utils.data.DataLoader(
  test_dataset,
  batch_size=args.batchsize,
  num_workers=0,
  shuffle=shuffle
)

### construct controlled set
extra_df = pd.read_csv(os.path.join(dataset_path,'csv/extra.csv'))
extra_dataset = dataset.InpaintingDataset(dataset_path,dataframe=extra_df,
                                  transform=transforms.Compose([
                          transforms.Resize(image_target_size,),
                          transforms.ToTensor()]))
extra_loader = torch.utils.data.DataLoader(
  extra_dataset,
  batch_size=30,
  num_workers=0,
  shuffle=False
)

sample_test_images = next(iter(extra_loader))

### populate ground truths
print('Populating groundtruths')

print('Train')
save_dir = f'tmp/train_groundtruths/'
if not os.path.exists(save_dir):
  os.makedirs(save_dir)

for index,(ground, mask,_) in enumerate(train_loader):
  for c,g in enumerate(ground):
    im = (g[0].numpy()*255).astype(np.uint8)
    im = grey2rgb(im)
    io.imsave(os.path.join(save_dir,f'batch_{index+1}_{c+1}.jpg'),im)

print('Test')
save_dir = f'tmp/test_groundtruths/'
if not os.path.exists(save_dir):
  os.makedirs(save_dir)

for index,(ground, mask,_) in enumerate(test_loader):
  for c,g in enumerate(ground):
    im = (g[0].numpy()*255).astype(np.uint8)
    im = grey2rgb(im)
    io.imsave(os.path.join(save_dir,f'batch_{index+1}_{c+1}.jpg'),im)

print('Controlled')
save_dir = f'tmp/extra_groundtruths/'
if not os.path.exists(save_dir):
  os.makedirs(save_dir)

for index,(ground, mask,_) in enumerate(extra_loader):
  for c,g in enumerate(ground):
    im = (g[0].numpy()*255).astype(np.uint8)
    im = grey2rgb(im)
    io.imsave(os.path.join(save_dir,f'batch_{index+1}_{c+1}.jpg'),im)

### end populate groundtruths

loaders = {
  'train': train_loader,
  'test': test_loader,
  'extra': extra_loader
}

### calculate fid statistics on groundtruths
print('Calculating FID statistics')

block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
inception_model = InceptionV3([block_idx])
inception_model = inception_model.to(device)
inception_model.eval()

train_fid_stats = -1
test_fid_stats = -1
if not debug:
  train_fid_stats = fid._compute_statistics_of_path('tmp/train_groundtruths',inception_model,50,2048,True)
  test_fid_stats = fid._compute_statistics_of_path('tmp/test_groundtruths',inception_model,50,2048,True)

### segmentation model
segment_model_state = torch.load(os.path.join('_states/face_segmentation/face_segment_model_ep200.pt'),map_location='cuda:0' if torch.cuda.is_available() else 'cpu')
n_classes = 4
segment_model = networks.UnetGenerator(1, n_classes, 7, ngf=32, norm_layer=functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True), 
                                use_dropout='False')
segment_model.load_state_dict(state)
segment_model = segment_model.to(device)
segment_model.eval()

state['train_fid'] = train_fid_stats
state['test_fid'] = test_fid_stats
state['inception_model'] = inception_model
state['segmentation_model'] = segmentation_model

print('Begin experiments')
for module in args.experiments:
  print(module)
  exp = importlib.import_module('experiment_list.{}'.format(module))
  exp.begin(state, loaders)
print('Finished~')