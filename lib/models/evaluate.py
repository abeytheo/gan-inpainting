import os
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import nn
import glob

from skimage.color import grey2rgb
import skimage.io as io
import skimage

import lib.models.loss as loss
import lib.pytorch_ssim as pytorch_ssim
import lib.fid.fid_score as fid

num_disp = 20
def from_model_object(test,model,model_root,epoch,save=True,is_flip_mask=False):
	eval_path = os.path.join(model_root,'output_images')
	if not os.path.exists(eval_path):
		os.makedirs(eval_path)

	loaded_G = model

	ground = test[0]
	mask = test[1]
	if is_flip_mask:
		mask = (1-mask)
	masked = ground * (1-mask)
	out = loaded_G(masked.cuda())

	#fig, axs = plt.subplots(num_disp, 4, figsize=(10, 480))
	fig, axs = plt.subplots(num_disp, 3, figsize=(10, 40))
	for i in range(num_disp):
		gt1 = ground[i].cpu().numpy().transpose((1, 2, 0))
		masked1 = masked[i].cpu().numpy().transpose((1, 2, 0))
		out1 = out[i].cpu().detach().numpy().transpose((1, 2, 0))

		axs[i,0].imshow(np.array(gt1)[:,:,0],cmap='Greys_r')
		axs[i,0].axis('off')
		axs[i,1].imshow(np.array(masked1)[:,:,0],cmap='Greys_r')
		axs[i,1].axis('off')
		axs[i,2].imshow(np.array(out1)[:,:,0],cmap='Greys_r')
		axs[i,2].axis('off')
	
	plt.suptitle('Epoch ' + str(epoch),y=0.4)
	if save:
		plt.savefig(os.path.join(eval_path,'epoch{}.png'.format(epoch)),format='png',dpi=100)

def from_saved_obj(test,network_architecture,model_root,epoch_list=[],save=True,is_flip_mask=False):
	eval_path = os.path.join(model_root,'evaluate')
	if not os.path.exists(eval_path):
		os.makedirs(eval_path)
	
	for e in epoch_list:
		torch.cuda.empty_cache() 
		G_statedict = torch.load(os.path.join(model_root,'epoch{}_G.pt'.format(e)),map_location=device)
		
		loaded_G = network_architecture.cpu()
		loaded_G.load_state_dict(G_statedict)
		
		ground = test[0]
		mask = test[1]
		if is_flip_mask:
			mask = (1-mask)
		masked = ground * (1-mask)
		out = loaded_G(masked.cpu())

    #fig, axs = plt.subplots(num_disp, 4, figsize=(10, 480))
		fig, axs = plt.subplots(num_disp, 3, figsize=(10, 40))
		for i in range(num_disp):
			gt1 = ground[i].cpu().numpy().transpose((1, 2, 0))
			masked1 = masked[i].cpu().numpy().transpose((1, 2, 0))
			out1 = out[i].cpu().detach().numpy().transpose((1, 2, 0))
			
			axs[i,0].imshow(np.array(gt1)[:,:,0],cmap='Greys_r')
			axs[i,0].axis('off')
			axs[i,1].imshow(np.array(masked1)[:,:,0],cmap='Greys_r')
			axs[i,1].axis('off')
			axs[i,2].imshow(np.array(out1)[:,:,0],cmap='Greys_r')
			axs[i,2].axis('off')
		
		plt.suptitle('Epoch ' + str(e),y=0.4)
		if save:
			plt.savefig(os.path.join(model_root,'evaluate/result_epoch{}.png'.format(e)),format='png',dpi=100)

def calculate_metric(device,loader,net,fid_stats,mode,inception_model,is_flip_mask=False):
  ### mode = train or test
  recon_rmse_global = 0
  recon_l1_global = 0
  
	l1_criterion = nn.L1Loss()
  rmse_criterion = loss.RMSELoss()
  
  size = 0
  target_dir = f'tmp/{mode}_m'
  if not os.path.exists(target_dir):
    os.makedirs(target_dir)
  else:
    files = glob.glob(f'{target_dir}/*')
    for f in files:
        os.remove(f)

  with torch.no_grad():
    for index,(input,mask,_) in enumerate(loader):
      mask = torch.ceil(mask.to(device))
      input = input.to(device)
      m = mask
      if is_flip_mask:
        m = (1-mask)
      masked = input * (1-m)
      out = net(masked)
      out = out * m + masked
      
      recon_rmse_global += rmse_criterion(input,out).item()
			recon_l1_global +=  l1_criterion(input,out).item()
      size += out.shape[0]
      
      for c,g in enumerate(out):
        im = (g[0].detach().cpu().numpy()*255).astype(np.uint8)
        im = grey2rgb(im)
        io.imsave(os.path.join(target_dir,f'batch_{index+1}_{c+1}.jpg'),im)

  m2, s2 = fid._compute_statistics_of_path(target_dir,inception_model,50,2048,True)

  fid_score = fid.calculate_frechet_distance(fid_stats[0], fid_stats[1], m2, s2)

  metric = {
    'recon_rmse_global': recon_rmse_global / (index+1),
    'recon_l1_global': recon_l1_global / (index+1),
    'fid': fid_score
  }
  return metric