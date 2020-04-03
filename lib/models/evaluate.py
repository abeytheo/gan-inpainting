import os
import numpy as np
from matplotlib import pyplot as plt
import torch

num_disp = 20
def from_model_object(test,model,model_root,epoch,save=True):
	eval_path = os.path.join(model_root,'output_images')
	if not os.path.exists(eval_path):
		os.makedirs(eval_path)

	loaded_G = model

	ground = test[0]
	mask = test[1]
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

def from_saved_obj(test,network_architecture,model_root,epoch_list=[],save=True):
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

def calculate_metric(loader,net,is_flip_mask=False):
	ssim = 0
	rmse_local = 0
	rmse_global = 0

	rmse_criterion = loss.RMSELoss()
	size = 0

	for input,mask,_ in loader:
		m = mask
		if is_flip_mask:
			m = (1-mask)
		masked = input * (1-m)
		out = net(masked)

		ssim += pytorch_ssim.ssim(input, out).item() * out.shape[0]
		rmse_global += rmse_criterion(input,out).item() * out.shape[0]
		rmse_local += rmse_criterion(input*m,out*m).item() * out.shape[0]

		size += out.shape[0]

	metric = {
		'ssim': ssim / size,
		'rmse_global': rmse_global / size,
		'rmse_local': rmse_local / size
	}

	return metric