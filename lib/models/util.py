
class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count =0

	def update(self, val, n=1):
		self.val = val
		self.sum += val*n
		self.count += n
		self.avg = self.sum / self.count
    
def set_requires_grad(nets,requires_grad):  
  for net in nets:
    for param in net.parameters():
      param.requires_grad = requires_grad

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)