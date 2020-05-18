import torch
import pandas as pd
import time
import pickle
import itertools
from torchvision import datasets, transforms
from torch import optim, nn
import os, logging, argparse

import lib.pytorch_ssim as pytorch_ssim
from lib.models import networks, util, loss, evaluate
from lib.data import dataset

### Parse arguments
parser = argparse.ArgumentParser(description='Training configurations')

parser.add_argument('-t','--title', type=str, help='Experiment title', required=True)
parser.add_argument('-e','--numepoch', nargs='?', type=int, default=1000, help='Number of training epoch')
parser.add_argument('-b','--batchsize', nargs='?', type=int, default=128, help='Batch size in one training epoch')
parser.add_argument('-g','--generator', nargs='?', type=str, choices=['unet','vgg19'], default='unet', help='Generator network name')
parser.add_argument('-d','--discriminator', nargs='?', type=str, choices=['patchgan','dcgan'], default='patchgan',help='Discriminator network name')
parser.add_argument('--imagedim', nargs='?', type=int, default=128,help='Image dimension')
parser.add_argument('--saveevery', nargs='?', type=int, default=50,help='Save network every N epoch(s)')
parser.add_argument('--updatediscevery', nargs='?', type=int, default=3,help='Backprop discriminator every N epoch(s)')
parser.add_argument('--evalevery', nargs='?', type=int, default=10,help='Evaluate test set every')
parser.add_argument('--lambda1', nargs='?', type=int, default=300,help='Hyperparameter lambda 1')
parser.add_argument('--lambda2', nargs='?', type=int, default=300,help='Hyperparameter lambda 2')

args = parser.parse_args()

### Create save directory
experiment_dir = "/home/s2125048/thesis/model/{}/".format(args.title)

if not os.path.exists(experiment_dir):
  os.makedirs(experiment_dir)
else:
  raise Exception('Experiment title has been used, please supply different title')

### Setup logger
logger = logging.getLogger('inpainting')
hdlr = logging.FileHandler('log/{}.log'.format(args.title))
formatter = logging.Formatter('%(asctime)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.INFO)

### settings
image_target_size = (args.imagedim,args.imagedim)
batch_size = args.batchsize
shuffle = True 

update_d_every = args.updatediscevery
evaluate_every = args.evalevery
num_epochs = args.numepoch
save_every = args.saveevery

lambda1=args.lambda1
lambda2=args.lambda2

use_cuda = True

logger.info(str(args))

device = torch.device("cuda:0")
logger.info('using device',device)

dataset_path = "/home/s2125048/thesis/dataset/"

### construct training dataset
train_df = pd.read_csv('train.csv')

train_dataset = dataset.InpaintingDataset(dataset_path,dataframe=train_df,
                                  transform=transforms.Compose([
                          transforms.Resize(image_target_size,),
                          transforms.ToTensor()]))

train_loader = torch.utils.data.DataLoader(
  train_dataset,
  batch_size=batch_size,
  num_workers=0,
  shuffle=shuffle
)

### construct test dataset
test_df = pd.read_csv('test.csv')

test_dataset = dataset.InpaintingDataset(dataset_path,dataframe=test_df,
                                  transform=transforms.Compose([
                          transforms.Resize(image_target_size,),
                          transforms.ToTensor()]))
test_loader = torch.utils.data.DataLoader(
  test_dataset,
  batch_size=batch_size,
  num_workers=0,
  shuffle=True
)

### construct extra dataset
extra_df = pd.read_csv('extra.csv')
extra_dataset = dataset.InpaintingDataset(dataset_path,dataframe=extra_df,
                                  transform=transforms.Compose([
                          transforms.Resize(image_target_size,),
                          transforms.ToTensor()]))
extra_loader = torch.utils.data.DataLoader(
  extra_dataset,
  batch_size=30,
  num_workers=0,
  shuffle=True
)

sample_test_images = next(iter(extra_loader))

### networks
net_G = networks.get_network('generator',args.generator).to(device)
net_D_global = networks.get_network('discriminator',args.discriminator).to(device)

rmse_criterion = loss.RMSELoss()
bce_criterion = nn.BCELoss()

G_optimizer = optim.Adam(net_G.parameters(), lr=0.0002, betas=(0.5, 0.999))
D_optimizer = optim.Adam(net_D_global.parameters(), lr=0.0002, betas=(0.5, 0.999))

training_epoc_hist = []
eval_hist = []

for epoch in range(num_epochs + 1):
  start = time.time()

  epoch_g_loss = {
    'total': 0,
    'adv': 0,
    'rmse': 0,
    'ssim': 0
  }

  epoch_d_loss = {
    'total': 0,
    'adv_real': 0,
    'adv_fake': 0
  }
  
  for i,(ground,mask,_) in enumerate(train_loader):

    ground = ground.to(device)
    mask = mask.to(device)

    masked = ground * (1-mask)

    ###
    # 1: Generator maximize [ log(D(G(x))) ]
    ###

    util.set_requires_grad([net_D_global],False)
    G_optimizer.zero_grad()

    ## Inpaint masked images
    inpainted = net_G(masked)

    d_pred_fake_global = net_D_global(inpainted).view(-1)

    ### we want the generator to be able to fool the discriminator, 
    ### thus, the goal is to enable the discriminator to always predict the inpainted images as real
    ### 1 = real, 0 = fake
    g_adv_loss_global = bce_criterion(d_pred_fake_global, torch.ones(len(d_pred_fake_global)).to(device))
    
    ### the inpainted image should be close to ground truth
    global_recon_loss = rmse_criterion(ground,inpainted)

    g_loss = g_adv_loss_global + lambda1 * global_recon_loss

    g_loss.backward()
    G_optimizer.step()
    
    ###
    # 2: Discriminator maximize [ log(D(x)) + log(1 - D(G(x))) ]
    ###
     # Update Discriminator networks.
    util.set_requires_grad([net_D_global],True)
    D_optimizer.zero_grad()

    ## We want the discriminator to be able to identify fake and real images
    d_pred_real_global = net_D_global(ground).view(-1)
    d_adv_loss_real_global = bce_criterion(d_pred_real_global, torch.ones(len(d_pred_real_global)).to(device) )
    
    d_pred_fake_global = net_D_global(inpainted.detach()).view(-1)
    d_adv_loss_fake_global = bce_criterion(d_pred_fake_global, torch.zeros(len(d_pred_fake_global)).to(device) )    

    d_loss = d_adv_loss_real_global + d_adv_loss_fake_global

    #if i % update_d_every == 0 and i > 0:
    d_loss.backward()
    D_optimizer.step()

    ### later will be divided by amount of training set
    ### in a minibatch, number of output may differ
    epoch_g_loss['total'] += g_loss.item()
    epoch_g_loss['rmse'] += global_recon_loss.item()
    epoch_g_loss['adv'] += g_adv_loss_global.item()
   
    ### calculate SSIM in training
    epoch_g_loss['ssim'] += pytorch_ssim.ssim(ground.detach(),inpainted.detach()).item()
    
    epoch_d_loss['total'] += d_loss.item()
    epoch_d_loss['adv_real'] += d_adv_loss_real_global.item()
    epoch_d_loss['adv_fake'] += d_adv_loss_fake_global.item()
  
  ### get epoch loss for G and D
  epoch_g_loss['total'] = epoch_g_loss['total'] / (i+1)
  epoch_g_loss['adv'] = epoch_g_loss['adv'] / (i+1)
  epoch_g_loss['rmse'] = epoch_g_loss['rmse'] / (i+1)
  epoch_g_loss['ssim'] = epoch_g_loss['ssim'] / (i+1)

  epoch_d_loss['adv_real'] = epoch_d_loss['adv_real'] / (i+1)
  epoch_d_loss['adv_fake'] = epoch_d_loss['adv_fake'] / (i+1)
  epoch_d_loss['total'] = epoch_d_loss / (i+1)
  
  training_epoc_hist.append({
    "g": epoch_g_loss,
    "d": epoch_d_loss
  })

  if epoch % evaluate_every == 0 and epoch > 0:
    test_metric = evaluate.calculate_metric(test_loader,net_G)
    extra_metric = evaluate.calculate_metric(extra_loader,net_G,is_flip_mask=True)
    eval_hist.append({
      'test': test_metric,
      'extra': extra_metric
    })
  
  if epoch % save_every == 0 and epoch > 0:
    torch.save(net_G.state_dict(), os.path.join(experiment_dir, "epoch{}_G.pt".format(epoch)))
    
    ### save training history
    with open(os.path.join(experiment_dir,'training_epoch_history.obj'),'wb') as handle:
      pickle.dump(training_epoc_hist, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    ### save evaluation history
    with open(os.path.join(experiment_dir,'eval_history.obj'),'wb') as handle:
      pickle.dump(eval_hist, handle, protocol=pickle.HIGHEST_PROTOCOL)

    evaluate.from_model_object(sample_test_images,net_G,experiment_dir,epoch)
    
  elapsed = time.time() - start
  logger.info(f'epoch: {epoch}, time: {elapsed:.3f}s, D loss: {d_loss.item():.3f}, G loss: {g_loss.item():.3f}')

with open(os.path.join(experiment_dir,'training_configuration.obj'),'wb') as handle:
      pickle.dump(args, handle, protocol=pickle.HIGHEST_PROTOCOL)