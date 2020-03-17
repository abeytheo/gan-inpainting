

def set_requires_grad(nets,requires_grad):  
  for net in nets:
    for param in net.parameters():
      param.requires_grad = requires_grad

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)