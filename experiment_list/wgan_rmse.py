import torch
import pandas as pd
import time, math
import pickle
import itertools
from torchvision import datasets, transforms
from torch import optim, nn
import os, logging, argparse
import copy

import lib.pytorch_ssim as pytorch_ssim
from lib.models import networks, util, loss, evaluate
from lib.data import dataset

def begin(state, loaders):
  ### EXPERIMENT CONFIGURATION
  state = state.copy()
  state.update(
    {
      'title': 'wgan_rmse'
    }
  )

  train_loader, test_loader, extra_loader = loaders['train'], loaders['test'], loaders['extra']

  ### Create save directory
  experiment_dir = "/home/s2125048/thesis/model/{}/".format(state['title'])

  if not os.path.exists(experiment_dir):
    os.makedirs(experiment_dir)

  ### Setup logger
  logger = logging.getLogger(state['title'])
  hdlr = logging.FileHandler('log/{}.log'.format(state['title']))
  formatter = logging.Formatter('%(asctime)s %(message)s')
  hdlr.setFormatter(formatter)
  logger.addHandler(hdlr)
  logger.setLevel(logging.INFO)

  ### settings
  image_target_size = (state['imagedim'],state['imagedim'])
  batch_size = state['batchsize']
  shuffle = True 

  update_d_every = state['updatediscevery']
  evaluate_every = state['evalevery']
  num_epochs = state['numepoch']
  save_every = state['saveevery']

  device = torch.device('cpu')
  if torch.cuda.is_available():
    device = torch.device("cuda:0")
    
  logger.info('using device',device)

  ### networks
  net_G = networks.get_network('generator',state['generator']).to(device)
  net_D_global = networks.PatchGANDiscriminator(sigmoid=False).to(device)

  rmse_criterion = loss.RMSELoss()
  bce_criterion = nn.BCELoss()
  l1_criterion = nn.L1Loss()

  G_optimizer = optim.RMSprop(net_G.parameters(),lr=0.00005)
  D_optimizer = optim.RMSprop(net_D_global.parameters(),lr=0.00005)

  update_g_every = 5

  training_epoc_hist = []
  eval_hist = []

  ### Wassterstein Loss

  one = torch.FloatTensor(1)
  mone = one * -1

  one = one.to(device)
  mone = mone.to(device)

  G_iter_count = 0
  D_iter_count = 0

  for epoch in range(num_epochs + 1):
    start = time.time()

    epoch_g_loss = {
      'total': 0,
      'recon': 0,
      'adv': 0,
      'update_count': 0
    }
    epoch_d_loss = {
        'total': 0,
        'adv_real': 0,
        'adv_fake': 0,
        'update_count': 0
    }
    gradient_hist = {
        'avg_g': {},
        'avg_d': {},
    }

    for n, p in net_G.named_parameters():
      if(("bias" not in n):
          gradient_hist['avg_g'][n] = 0
    for n, p in net_D_global.named_parameters():
      if("bias" not in n):
          gradient_hist['avg_d'][n] = 0
    
    for current_batch_index,(ground,mask,_) in enumerate(train_loader):

      curr_batch_size = ground.shape[0]
      ground = ground.to(device)
      mask = torch.ceil(mask.to(device))

      ## Inpaint masked images
      masked = ground * (1-mask)

      inpainted = net_G(masked)

      ### only replace the masked area
      inpainted = masked + inpainted * mask

      ###
      # 1: Discriminator maximize [ log(D(x)) + log(1 - D(G(x))) ]
      ###
      # Update Discriminator networks.
      util.set_requires_grad([net_D_global],True)

      D_optimizer.zero_grad()

      ## We want the discriminator to be able to identify fake and real images+ add_label_noise(noise_std,curr_batch_size)

      d_pred_real = net_D_global(ground)
      d_pred_fake = net_D_global(inpainted.detach())

      d_loss_real = torch.mean(d_pred_real).view(1)
      d_loss_real.backward(one)

      d_loss_fake = torch.mean(d_pred_fake).view(1)
      d_loss_fake.backward(mone)

      d_loss =  d_loss_real - d_loss_fake

      D_G_z1 = d_pred_fake.mean().item()
      D_x = d_pred_real.mean().item()
      D_optimizer.step()

      D_iter_count +=1

      # Clip weights of discriminator
      for p in net_D_global.parameters():
          p.data.clamp_(-0.01, 0.01)
      
      ### On initial training epoch, we want D to converge as fast as possible before updating the generator
      ### that's why, G is updated every 100 D iterations
      if G_iter_count < 25 or G_iter_count % 500 == 0:
        update_G_every_batch = 140
      else:
        update_G_every_batch = update_g_every

      ### Update generator every `D_iter`
      if current_batch_index % update_G_every_batch == 0 and current_batch_index > 0:

        ###
        # 2: Generator maximize [ log(D(G(x))) ]
        ###

        util.set_requires_grad([net_D_global],False)
        G_optimizer.zero_grad()

        d_pred_fake = net_D_global(inpainted).view(-1)

        ### we want the generator to be able to fool the discriminator, 
        ### thus, the goal is to enable the discriminator to always predict the inpainted images as real
        ### 1 = real, 0 = fake
        g_adv_loss = torch.mean(d_pred_fake).view(1)

        ### the inpainted image should be close to ground truth
        recon_loss = rmse_criterion(ground,inpainted)

        g_loss = g_adv_loss + recon_loss
        g_loss.backward()

        D_G_z2 = d_pred_fake.mean().item()
        G_optimizer.step()

        G_iter_count +=1 

        epoch_g_loss['total'] += g_loss.item()
        epoch_g_loss['recon'] += recon_loss.item()
        epoch_g_loss['adv'] += g_adv_loss.item()
        epoch_g_loss['update_count'] += 1

        for n, p in net_G.named_parameters():
          if("bias" not in n):
            gradient_hist['avg_g'][n] += p.grad.abs().mean().item()

        logger.info('[epoch %d/%d][batch %d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                % (epoch, num_epochs, current_batch_index, len(train_loader),
                    d_loss.item(), g_adv_loss.item()))
        
      ### later will be divided by amount of training set
      ### in a minibatch, number of output may differ
      epoch_d_loss['total'] += d_loss.item()
      epoch_d_loss['adv_real'] += d_loss_real.item()
      epoch_d_loss['adv_fake'] += d_loss_fake.item()
      epoch_d_loss['update_count'] += 1

      for n, p in net_D_global.named_parameters():
        if("bias" not in n):
          gradient_hist['avg_d'][n] += p.grad.abs().mean().item()
    
    ### get gradient and epoch loss for generator
    try:
      epoch_g_loss['total'] = epoch_g_loss['total'] / epoch_g_loss['update_count']
      epoch_g_loss['recon'] = epoch_g_loss['recon'] / epoch_g_loss['update_count']
      epoch_g_loss['adv'] = epoch_g_loss['adv'] / epoch_g_loss['update_count']

      for n, p in net_G.named_parameters():
        if("bias" not in n):
          gradient_hist['avg_g'][n] = gradient_hist['avg_g'][n] / epoch_g_loss['update_count']

    except:

      ### set invalid values when G is not updated
      epoch_g_loss['total'] = -7777
      epoch_g_loss['recon'] = -7777
      epoch_g_loss['adv'] = -7777

      for n, p in net_G.named_parameters():
        if("bias" not in n):
          gradient_hist['avg_g'][n] = -7777

    
    ### get gradient and epoch loss for discriminator
    epoch_d_loss['total'] = epoch_d_loss['total'] / epoch_d_loss['update_count']
    epoch_d_loss['adv_real'] = epoch_d_loss['adv_real'] / epoch_d_loss['update_count']
    epoch_d_loss['adv_fake'] = epoch_d_loss['adv_fake'] / epoch_d_loss['update_count']
    
    for n, p in net_D_global.named_parameters():
      if("bias" not in n):
        gradient_hist['avg_d'][n] = gradient_hist['avg_d'][n] / epoch_d_loss['update_count']
    
    training_epoc_hist.append({
      "g": epoch_g_loss,
      "d": epoch_d_loss,
      "gradients": gradient_hist
    })

    if epoch % evaluate_every == 0 and epoch > 0:
      train_metric = evaluate.calculate_metric(device,train_loader,net_G,state['train_fid'],mode='train',inception_model=state['inception_model'])
      test_metric = evaluate.calculate_metric(device,test_loader,net_G,state['test_fid'],mode='test',inception_model=state['inception_model'])
      eval_hist.append({
        'train': train_metric,
        'test': test_metric
      })

      logger.info("VALIDATION: train - {}, test - {}".format(str(train_metric),str(test_metric)))

      ### save training history
      with open(os.path.join(experiment_dir,'training_epoch_history.obj'),'wb') as handle:
        pickle.dump(training_epoc_hist, handle, protocol=pickle.HIGHEST_PROTOCOL)
      
      ### save evaluation history
      with open(os.path.join(experiment_dir,'eval_history.obj'),'wb') as handle:
        pickle.dump(eval_hist, handle, protocol=pickle.HIGHEST_PROTOCOL)
  
    if epoch % save_every == 0 and epoch > 0:
      try:
        torch.save(net_G.state_dict(), os.path.join(experiment_dir, "epoch{}_G.pt".format(epoch)))
      except:
        logger.error(traceback.format_exc())
        pass
        
    elapsed = time.time() - start
    logger.info(f'epoch: {epoch}, time: {elapsed:.3f}s, D total: {epoch_d_loss["total"]:.10f}, D real: {epoch_d_loss["adv_real"]:.10f}, D fake: {epoch_d_loss["adv_fake"]:.10f}, G total: {epoch_g_loss["total"]:.3f}, G adv: {epoch_g_loss["adv"]:.10f}, G recon: {epoch_g_loss["recon"]:.3f}')