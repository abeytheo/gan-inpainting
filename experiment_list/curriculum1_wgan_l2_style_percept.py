import torch
import pandas as pd
import time, math
import pickle
import itertools, functools
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
      'title': 'curriculum1_wgan_style_percept'
    }
  )

  train_loader, test_loader, extra_loader = loaders['train'], loaders['test'], loaders['extra']

  ### Create save directory
  experiment_dir = "/home/s2125048/thesis/model/{}/{}/".format(state['execution_date'],state['title'])
  log_dir = os.path.join(experiment_dir,'log')

  if not os.path.exists(experiment_dir):
    os.makedirs(experiment_dir)
    os.makedirs(log_dir)

  ### Setup logger
  logger = logging.getLogger(state['title'])
  hdlr = logging.FileHandler('{}/{}.log'.format(log_dir,state['title']))
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

  ### networks
  net_G = networks.get_network('generator',state['generator']).to(device)
  net_D_global = networks.PatchGANDiscriminator(sigmoid=False).to(device)
  segment_model = state['segmentation_model']

  G_state = torch.load('/home/s2125048/thesis/model/20200707_220810/1_curriculum_wgan_l2/epoch80_G.pt',map_location='cuda:0' if torch.cuda.is_available() else 'cpu')
  net_G.load_state_dict(G_state)
  net_G.train()

  D_state = torch.load('/home/s2125048/thesis/model/20200707_220810/1_curriculum_wgan_l2/epoch80_D.pt',map_location='cuda:0' if torch.cuda.is_available() else 'cpu')
  net_D_global.load_state_dict(D_state)
  net_D_global.train()

  ### criterions
  rmse_global_criterion = loss.RMSELoss()
  rmse_local_criterion = loss.LocalLoss(loss.RMSELoss())
  bce_criterion = nn.BCELoss()
  l1_criterion = nn.L1Loss()

  w = torch.tensor([0,1.2,0.7,0.7])
  weight_ce_criterion = nn.CrossEntropyLoss(weight=w).to(device)

  ### optimizer
  G_optimizer = optim.RMSprop(net_G.parameters(),lr=0.00005)
  D_optimizer = optim.RMSprop(net_D_global.parameters(),lr=0.00005)

  update_g_every = 5

  unique_labels = [0,1,2,3]
  training_epoc_hist = []
  eval_hist = []

  ### Wassterstein Loss
  
  one = torch.FloatTensor(1)
  mone = one * -1

  one = one.to(device)
  mone = mone.to(device)

  G_iter_count = 0
  D_iter_count = 0
  
  lowest_fid = {
      'value': 1e9,
      'epoch': 0
  }

  for epoch in range(num_epochs + 1):
    start = time.time()

    epoch_g_loss = {
      'total': 0,
      'recon_global': 0,
      'recon_local': 0,
      'face_parsing': 0,
      'perceptual': 0,
      'style': 0,
      'tv': 0,
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

    ### face attr segmentation metric
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

    for n, p in net_G.named_parameters():
      if("bias" not in n):
          gradient_hist['avg_g'][n] = 0
    for n, p in net_D_global.named_parameters():
      if("bias" not in n):
          gradient_hist['avg_d'][n] = 0
    
    for current_batch_index,(ground,mask,segment) in enumerate(train_loader):

      ## Freeze G while D is learning
      util.set_requires_grad([net_D_global],True)
      net_G.eval()

      curr_batch_size = ground.shape[0]
      ground = ground.to(device)
      segment = segment.to(device)
      mask = torch.ceil(mask.to(device))

      ## Inpaint masked images
      masked = ground * (1-mask)

      out = net_G(masked)

      ### only replace the masked area
      inpainted = masked + out * mask

      ###
      # 1: Discriminator maximize [ log(D(x)) + log(1 - D(G(x))) ]
      ###
      # Update Discriminator networks.

      D_optimizer.zero_grad()

      ## We want the discriminator to be able to identify fake and real images+ add_label_noise(noise_std,curr_batch_size)

      d_pred_real = net_D_global(ground)
      d_pred_fake = net_D_global(inpainted.detach())

      d_loss_real = torch.mean(d_pred_real).view(1)
      d_loss_real.backward(one)

      d_loss_fake = torch.mean(d_pred_fake).view(1)
      d_loss_fake.backward(mone)

      ### put absolute value for EM distance
      d_loss =  torch.abs(d_loss_real - d_loss_fake)

      D_G_z1 = d_pred_fake.mean().item()
      D_x = d_pred_real.mean().item()
      D_optimizer.step()

      D_iter_count +=1

      # Clip weights of discriminator
      for p in net_D_global.parameters():
          p.data.clamp_(-0.01, 0.01)
      
      ### On initial training epoch, we want D to converge as fast as possible before updating the generator
      ### that's why, G is updated every 100 D iterations
      # if G_iter_count < 25 or G_iter_count % 500 == 0:
      #   update_G_every_batch = 140
      # else:
      update_G_every_batch = update_g_every

      ### Update generator every `D_iter`
      if current_batch_index % update_G_every_batch == 0 and current_batch_index > 0:

        ###
        # 2: Generator maximize [ log(D(G(x))) ]
        ###

        ## Freeze D while G is learning
        util.set_requires_grad([net_D_global],False)
        net_G.train()

        G_optimizer.zero_grad()

        d_pred_fake = net_D_global(inpainted).view(-1)

        ### we want the generator to be able to fool the discriminator, 
        ### thus, the goal is to enable the discriminator to always predict the inpainted images as real
        ### 1 = real, 0 = fake
        g_adv_loss = torch.mean(d_pred_fake).view(1)

        ### the inpainted image should be close to ground truth
        recon_global_loss = rmse_global_criterion(ground,out)
        recon_local_loss = rmse_local_criterion(ground,out,mask)

        ### face parsing loss
        inpainted_segment = segment_model(out)
        # g_face_parsing_loss = 0.1 * weight_ce_criterion(inpainted_segment,segment)

        ### perceptual and style
        # g_perceptual_loss_comp = loss.perceptual_loss(inpainted,ground,weight=0.01)
        # g_perceptual_loss_out = loss.perceptual_loss(out,ground,weight=0.011)
        # g_style_loss_comp = loss.style_loss(inpainted,ground,weight_s=0.1)
        # g_style_loss_out = loss.style_loss(out,ground,weight_s=0.1)

        g_perceptual_loss_comp, g_style_loss_comp = loss.perceptual_and_style_loss(inpainted,ground,weight_p=0.01,weight_s=0.1)
        g_perceptual_loss_out, g_style_loss_out = loss.perceptual_and_style_loss(out,ground,weight_p=0.01,weight_s=0.1)

        g_perceptual_loss = g_perceptual_loss_comp + g_perceptual_loss_out
        g_style_loss = g_style_loss_comp + g_style_loss_out

        ### tv
        # g_tv_loss_comp = loss.tv_loss(inpainted,tv_weight=1)
        # g_tv_loss_out = loss.tv_loss(out,tv_weight=1)
        # g_tv_loss = g_tv_loss_comp + g_tv_loss_out

        g_loss = g_adv_loss + recon_global_loss + 10 * recon_local_loss + g_perceptual_loss + g_style_loss
        # g_loss = g_adv_loss + recon_global_loss + 5 * recon_local_loss + \
        #          g_perceptual_loss + \
        #          g_style_loss + \
        #          g_tv_loss + \
        #          g_face_parsing_loss

        g_loss.backward()

        D_G_z2 = d_pred_fake.mean().item()
        G_optimizer.step()

        G_iter_count +=1 

        ### update segmentation metric
        class_m, across_class_m = evaluate.calculate_segmentation_eval_metric(segment,inpainted_segment,unique_labels)
        for u in unique_labels:
          for k, _ in classes_metric[u].items():
            classes_metric[u][k].update(class_m[u][k],ground.size(0))
        
        for k, _ in across_class_metric.items():
          across_class_metric[k].update(across_class_m[k], ground.size(0))

        epoch_g_loss['total'] += g_loss.item()
        epoch_g_loss['recon_global'] += recon_global_loss.item()
        epoch_g_loss['recon_local'] += recon_local_loss.item()
        epoch_g_loss['adv'] += g_adv_loss.item()
        # epoch_g_loss['tv'] += g_tv_loss.item()
        epoch_g_loss['perceptual'] += g_perceptual_loss.item()
        epoch_g_loss['style'] += g_style_loss.item()
        # epoch_g_loss['face_parsing'] += g_face_parsing_loss.item()
        epoch_g_loss['update_count'] += 1

        for n, p in net_G.named_parameters():
          if("bias" not in n):
            gradient_hist['avg_g'][n] += p.grad.abs().mean().item()

        # logger.info('[epoch %d/%d][batch %d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
        #         % (epoch, num_epochs, current_batch_index, len(train_loader),
        #             d_loss.item(), g_adv_loss.item()))
        
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
      epoch_g_loss['adv'] = epoch_g_loss['adv'] / epoch_g_loss['update_count']
      epoch_g_loss['recon_global'] = epoch_g_loss['recon_global'] / epoch_g_loss['update_count']
      epoch_g_loss['recon_local'] = epoch_g_loss['recon_local'] / epoch_g_loss['update_count']
      # epoch_g_loss['tv'] = epoch_g_loss['tv'] / epoch_g_loss['update_count']
      epoch_g_loss['perceptual'] = epoch_g_loss['perceptual'] / epoch_g_loss['update_count']
      epoch_g_loss['style'] = epoch_g_loss['style'] / epoch_g_loss['update_count']
      # epoch_g_loss['face_parsing'] = epoch_g_loss['face_parsing'] / epoch_g_loss['update_count']
      
      for n, p in net_G.named_parameters():
        if("bias" not in n):
          gradient_hist['avg_g'][n] = gradient_hist['avg_g'][n] / epoch_g_loss['update_count']
    except:
      pass
    
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

    is_alltime_low = False

    if epoch % evaluate_every == 0:
      train_metric = evaluate.calculate_metric(device,train_loader,net_G,state['train_fid'],mode='train',inception_model=state['inception_model'],epoch=epoch,segment_model=state['segmentation_model'])
      test_metric = evaluate.calculate_metric(device,test_loader,net_G,state['test_fid'],mode='test',inception_model=state['inception_model'],epoch=epoch,segment_model=state['segmentation_model'])
      eval_hist.append({
        'train': train_metric,
        'test': test_metric
      })

      logger.info("----------")
      logger.info("Validation")
      logger.info("----------")
      logger.info("Train recon global: {glo: .4f}, local: {loc: .4f}, FID: {fid: .4f}".format(glo=train_metric['recon_rmse_global'],loc=train_metric['recon_rmse_local'],fid=train_metric['fid']))
      logger.info("Test recon global: {glo: .4f}, local: {loc: .4f}, FID: {fid: .4f}".format(glo=test_metric['recon_rmse_global'],loc=test_metric['recon_rmse_local'],fid=test_metric['fid']))
      for u in unique_labels:
        logger.info("Class {}: Precision {prec: .4f}, Recall {rec: .4f}, IoU {iou: .4f}".format(u,prec=test_metric['face_parsing_metric']['indv_class'][u]['precision'].avg,
                                                                                        rec=test_metric['face_parsing_metric']['indv_class'][u]['recall'].avg,
                                                                                        iou=test_metric['face_parsing_metric']['indv_class'][u]['iou'].avg))
      logger.info("Across Classes: Precision {prec.avg: .4f}, Recall {rec.avg: .4f}, IoU {iou.avg: .4f}".format(prec=test_metric['face_parsing_metric']['accross_class']['precision'],
                                                                    rec=test_metric['face_parsing_metric']['accross_class']['recall'],
                                                                    iou=test_metric['face_parsing_metric']['accross_class']['iou']))
                                                                    
      ### save training history
      with open(os.path.join(experiment_dir,'training_epoch_history.obj'),'wb') as handle:
        pickle.dump(training_epoc_hist, handle, protocol=pickle.HIGHEST_PROTOCOL)
      
      ### save evaluation history
      with open(os.path.join(experiment_dir,'eval_history.obj'),'wb') as handle:
        pickle.dump(eval_hist, handle, protocol=pickle.HIGHEST_PROTOCOL)

      if test_metric['fid'] < lowest_fid['value']:
        is_alltime_low = True
        logger.info('All time low test FID: {alltime} < {prev_fid} at epoch {prev_ep}'.format(alltime=test_metric['fid'],prev_fid=lowest_fid['value'],prev_ep=lowest_fid['epoch']))
        lowest_fid['value'] = test_metric['fid']
        lowest_fid['epoch'] = epoch
  
    if (epoch % save_every == 0 and epoch > 0) or (is_alltime_low):
      try:
        torch.save(net_G.state_dict(), os.path.join(experiment_dir, "epoch{}_G.pt".format(epoch)))
        if is_alltime_low:
          torch.save(net_D_global.state_dict(), os.path.join(experiment_dir, "epoch{}_D.pt".format(epoch)))
      except:
        logger.error(traceback.format_exc())
        pass
        
    elapsed = time.time() - start
    
    logger.info(f'Epoch: {epoch}, time: {elapsed:.3f}s')
    logger.info('')
    logger.info('> Adversarial')
    logger.info(f'EM Distance: {epoch_d_loss["total"]:.10f}')
    logger.info('> Reconstruction')
    logger.info(f'Generator recon global: {epoch_g_loss["recon_global"]:.5f}, local: {epoch_g_loss["recon_local"]:.5f}')
    logger.info('> Face Parsing')
    for u in unique_labels:
      logger.info("Class {}: Precision {prec: .4f}, Recall {rec: .4f}, IoU {iou: .4f}".format(u,prec=classes_metric[u]['precision'].avg,
                                                                                        rec=classes_metric[u]['recall'].avg,
                                                                                        iou=classes_metric[u]['iou'].avg))
    logger.info("Across Classes: Precision {prec.avg: .4f}, Recall {rec.avg: .4f}, IoU {iou.avg: .4f}".format(prec=across_class_metric['precision'],
                                                                    rec=across_class_metric['recall'],
                                                                    iou=across_class_metric['iou']))