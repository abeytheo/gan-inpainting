for epoch in range(num_epochs + 1):
  start = time.time()

  epoch_g_loss = {
    'total': 0,
    'rmse_global': 0,
    'adversarial': 0
  }
  epoch_d_loss = {
      'real': 0,
      'fake': 0,
      'total': 0
  }
  gradient_hist = {
      'avg_g': {},
      'max_g': {},
      'avg_d': {},
      'max_d': {}
  }
  for n, p in net_G.named_parameters():
    if(p.requires_grad) and ("bias" not in n):
        gradient_hist['avg_g'][n] = 0
        gradient_hist['max_g'][n] = 0
  for n, p in net_D_global.named_parameters():
    if(p.requires_grad) and ("bias" not in n):
        gradient_hist['avg_d'][n] = 0
        gradient_hist['max_d'][n] = 0
  
  for current_batch_index,(ground,mask,_) in enumerate(train_loader):

    curr_batch_size = ground.shape[0]
    ground = ground.to(device)
    mask = mask.to(device)

    masked = ground * (1-mask)

    ###
    # 1: Discriminator maximize [ log(D(x)) + log(1 - D(G(x))) ]
    ###
    # Update Discriminator networks.

    set_requires_grad([net_D_global],True)
    D_optimizer.zero_grad()

    ## We want the discriminator to be able to identify fake and real images+ add_label_noise(noise_std,curr_batch_size)
    # d_pred_real_global = net_D_global(noisy_ground).view(-1)
    # d_pred_fake_global = net_D_global(noisy_inpainted.detach()).view(-1)

    d_pred_real_global = net_D_global(ground).view(-1)
    d_adv_loss_real_global = bce_criterion(d_pred_real_global, torch.ones(len(d_pred_real_global)).to(device) )
    d_adv_loss_real_global.backward()

    D_x = d_adv_loss_real_global.mean().item()

    d_pred_fake_global = net_D_global(inpainted.detach()).view(-1)
    d_adv_loss_fake_global = bce_criterion(d_pred_fake_global, torch.zeros(len(d_pred_fake_global)).to(device) )    
    d_adv_loss_fake_global.backward()

    D_G_z1 = d_pred_fake_global.mean().item()

    #d_adv_loss_real_global = rmse_criterion(d_pred_real_global, torch.ones(len(d_pred_real_global)).to(device) + add_label_noise(noise_std * decay_factor, curr_batch_size) )
    #d_adv_loss_fake_global = rmse_criterion(d_pred_fake_global, torch.zeros(len(d_pred_fake_global)).to(device) + add_label_noise(noise_std * decay_factor, curr_batch_size) )    

    d_loss = d_adv_loss_real_global + d_adv_loss_fake_global

    D_optimizer.step()

    ###
    # 2: Generator maximize [ log(D(G(x))) ]
    ###

    set_requires_grad([net_D_global],False)
    G_optimizer.zero_grad()

    ## Inpaint masked images
    inpainted = net_G(masked)

    ## add gaussian noise to real and fake images
    # noisy_inpainted = add_noise(inpainted, noise_std*decay_factor, curr_batch_size)
    # noisy_ground = add_noise(ground, noise_std*decay_factor, curr_batch_size)

    #d_pred_fake_global = net_D_global(noisy_inpainted).view(-1)
    d_pred_fake_global = net_D_global(inpainted).view(-1)

    ### we want the generator to be able to fool the discriminator, 
    ### thus, the goal is to enable the discriminator to always predict the inpainted images as real
    ### 1 = real, 0 = fake
    g_adv_loss_global = bce_criterion(d_pred_fake_global, torch.ones(len(d_pred_fake_global)).to(device))

    ### the inpainted image should be close to ground truth
    
    global_recon_loss = rmse_criterion(ground,inpainted)
    #local_recon_loss = rmse_criterion(mask*ground,mask*inpainted)

    g_loss = g_adv_loss_global + lambda1 * global_recon_loss

    g_loss.backward()

    D_G_z2 = d_pred_fake_global.mean().item()
    G_optimizer.step()

    if current_batch_index % 10 == 0:
      print('[epoch %d/%d][batch %d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
            % (epoch, num_epochs, current_batch_index, len(train_loader),
                d_loss.item(), g_adv_loss_global.item(), D_x, D_G_z1, D_G_z2))

    ### later will be divided by amount of training set
    ### in a minibatch, number of output may differ
    epoch_d_loss['real'] += d_adv_loss_real_global.item()
    epoch_d_loss['fake'] += d_adv_loss_fake_global.item()
    epoch_d_loss['total'] += d_loss.item()

    for n, p in net_D_global.named_parameters():
      if(p.requires_grad) and ("bias" not in n):
        gradient_hist['avg_d'][n] += p.grad.abs().mean().item()
        gradient_hist['max_d'][n] += p.grad.abs().max().item()

    epoch_g_loss['total'] += g_loss.item()
    epoch_g_loss['rmse_global'] += global_recon_loss.item()
    epoch_g_loss['adversarial'] += g_adv_loss_global.item()

    for n, p in net_G.named_parameters():
      if(p.requires_grad) and ("bias" not in n):
        gradient_hist['avg_g'][n] += p.grad.abs().mean().item()
        gradient_hist['max_g'][n] += p.grad.abs().max().item()

  ### get epoch loss for G and D
  epoch_g_loss['total'] = epoch_g_loss['total'] / (current_batch_index + 1)
  epoch_g_loss['rmse_global'] = epoch_g_loss['rmse_global'] / (current_batch_index + 1)
  epoch_g_loss['adversarial'] = epoch_g_loss['adversarial'] / (current_batch_index + 1)

  epoch_d_loss['total'] = epoch_d_loss['total'] / (current_batch_index + 1)
  epoch_d_loss['real'] = epoch_d_loss['real'] / (current_batch_index + 1)
  epoch_d_loss['fake'] = epoch_d_loss['fake'] / (current_batch_index + 1)

  for n, p in net_G.named_parameters():
    if(p.requires_grad) and ("bias" not in n):
      gradient_hist['avg_g'][n] = gradient_hist['avg_g'][n] / (current_batch_index + 1)
      gradient_hist['max_g'][n] = gradient_hist['max_g'][n] / (current_batch_index + 1)
  
  for n, p in net_D_global.named_parameters():
    if(p.requires_grad) and ("bias" not in n):
      gradient_hist['avg_d'][n] = gradient_hist['avg_d'][n] / ((current_batch_index + 1)/update_d_every)
      gradient_hist['max_d'][n] = gradient_hist['max_d'][n] / ((current_batch_index + 1)/update_d_every)
  
  training_epoc_hist.append({
    "g": epoch_g_loss,
    "d": epoch_d_loss,
    "gradients": gradient_hist
  })

  if epoch % evaluate_every == 0 and epoch > 0:
    test_metric = calculate_metric(test_loader,net_G)
    eval_hist.append({
      'test': test_metric
    })

    ### save training history
    with open(os.path.join(experiment_dir,'training_epoch_history.obj'),'wb') as handle:
      pickle.dump(training_epoc_hist, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    ### save evaluation history
    with open(os.path.join(experiment_dir,'eval_history.obj'),'wb') as handle:
      pickle.dump(eval_hist, handle, protocol=pickle.HIGHEST_PROTOCOL)
 
    # plot_grad_flow(net_G.named_parameters(),'Generator')
    # plot_grad_flow(net_D_global.named_parameters(),'Discriminator')
  
  if epoch % save_every == 0 and epoch > 0:
    try:
      torch.save(net_G.state_dict(), os.path.join(experiment_dir, "epoch{}_G.pt".format(epoch)))
    except:
      print(traceback.format_exc())
      pass
  
  if epoch % decay_every == 0 and epoch > 0:
    decay_factor = decay_factor * decay_rate
       
  elapsed = time.time() - start
  print(f'epoch: {epoch}, time: {elapsed:.3f}s, D total: {epoch_d_loss["total"]:.3f}, D real: {epoch_d_loss["real"]:.3f}, D fake: {epoch_d_loss["fake"]:.3f}, G total: {epoch_g_loss["total"]:.3f}, G adv: {epoch_g_loss["adversarial"]:.3f}, G recon: {epoch_g_loss["rmse_global"]:.3f}')
