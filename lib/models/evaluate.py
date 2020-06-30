import os
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import nn
import glob

from skimage.color import grey2rgb
import skimage.io as io
import skimage

import lib.models.util as util
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

def calculate_metric(device,loader,net,fid_stats,mode,inception_model,epoch,segment_model,is_flip_mask=False):
  ### mode = train or test

  ### face segmentation metric
  unique_labels = [0,1,2,3]
  classes_metric = {}
  for u in unique_labels:
    classes_metric[u] = {
        'precision': util.AverageMeter(),
        'recall': util.AverageMeter(),
        'iou': util.AverageMeter()
    }
  across_class_metric = {
      'precision': util.AverageMeter(),
      'recall': util.AverageMeter(),
      'iou': util.AverageMeter()
  }

  ### recon
  recon_l1_global = 0
  recon_l1_local = 0
  recon_rmse_global = 0
  recon_rmse_local = 0

  ### perceptual
  perceptual_out = 0
  perceptual_comp = 0

  ### criterions
  l1_global_criterion = nn.L1Loss()
  rmse_global_criterion = loss.RMSELoss()

  l1_local_criterion = loss.LocalLoss(nn.L1Loss)
  rmse_local_criterion = loss.LocalLoss(loss.RMSELoss())

  size = 0
  target_dir = f'tmp/{mode}_m'
  if not os.path.exists(target_dir):
    os.makedirs(target_dir)
  else:
    files = glob.glob(f'{target_dir}/*')
    for f in files:
        os.remove(f)

  with torch.no_grad():
    for index,(input,mask,segment) in enumerate(loader):
      mask = torch.ceil(mask.to(device))
      segment = segment.to(device)
      input = input.to(device)
      m = mask
      if is_flip_mask:
        m = (1-mask)
      masked = input * (1-m)
      out = net(masked)
      inpainted = out * m + masked

      ### segmentation
      segment_prediction = segment_model(out)

      ### recon
      recon_rmse_global += rmse_global_criterion(input,out).item()
      recon_l1_global +=  l1_global_criterion(input,out).item()
      
      recon_rmse_local += rmse_local_criterion(input,out,m).item()
      recon_l1_local +=  l1_local_criterion(input,out,m).item()
      size += out.shape[0]

      ### content
      perceptual_comp += loss.perceptual_loss(inpainted,input,weight=1).item()
      perceptual_out += loss.perceptual_loss(out,input,weight=1).item()

      ### face parsing
      class_m, across_class_m = calculate_segmentation_eval_metric(segment,segment_prediction,unique_labels)
      for u in unique_labels:
        for k, _ in classes_metric[u].items():
          classes_metric[u][k].update(class_m[u][k],input.size(0))
      
      for k, _ in across_class_metric.items():
        across_class_metric[k].update(across_class_m[k], input.size(0))
      
      for c,g in enumerate(inpainted):
        im = (g[0].detach().cpu().numpy()*255).astype(np.uint8)
        im = grey2rgb(im)
        io.imsave(os.path.join(target_dir,f'batch_{index+1}_{c+1}.jpg'),im)

  fid_score = -1
  if fid_stats:
    m2, s2 = fid._compute_statistics_of_path(target_dir,inception_model,50,2048,True)
    fid_score = fid.calculate_frechet_distance(fid_stats[0], fid_stats[1], m2, s2)

  metric = {
    'recon_rmse_global': recon_rmse_global / (index+1),
    'recon_l1_global': recon_l1_global / (index+1),
    'recon_rmse_local': recon_rmse_local / (index+1),
    'recon_l1_local': recon_l1_local / (index+1),
    'perceptual_comp': perceptual_comp / (index+1),
    'perceptual_out': perceptual_out / (index+1),
    'face_parsing_metric': {
      'indv_class': classes_metric,
      'accross_class': across_class_metric
    },
    'fid': fid_score,
    'epoch': epoch
  }
  return metric

def calculate_segmentation_eval_metric(labels,outputs,unique_labels):

  prediction = torch.argmax(outputs,1)

  eps = 1e-32
  batch = labels.shape[0]
  metric = {}

  across_class_metric = {
      'precision': 0,
      'recall': 0,
      'iou': 0
  }

  for u in unique_labels:
    g = labels.detach().clone().view(batch,-1)
    p = prediction.detach().clone().view(batch,-1)

    g[g == u] = -1
    g[g != -1] = 0
    g[g == -1] = 1

    p[p == u] = -1
    p[p != -1] = 0
    p[p == -1] = 1

    intersection = (p & g).sum(1).float()
    union = (p | g).sum(1).float()
    iou = intersection / (union + eps)

    ### how many predicted pixels are true
    precision = ((intersection) / (p.sum(1) + eps))

    ### how many true pixels are returned
    recall = ((intersection) / (g.sum(1) + eps))

    metric[u] = {'precision': precision.mean(), 'recall': recall.mean(), 'iou': iou.mean()}
    across_class_metric['precision'] += precision.mean()
    across_class_metric['recall'] += recall.mean()
    across_class_metric['iou'] += iou.mean()
  
  across_class_metric['precision'] = across_class_metric['precision'] / len(unique_labels)
  across_class_metric['recall'] = across_class_metric['recall'] / len(unique_labels)
  across_class_metric['iou'] = across_class_metric['iou'] / len(unique_labels)

  return metric, across_class_metric