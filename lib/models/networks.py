import torch
from torch import nn

import functools

### Functions

class Identity(nn.Module):
    def forward(self, x):
        return x

def get_network(type,name):
  if type == 'generator':
    if name == 'vgg19':
      return VGG19Generator()
    elif name == 'unet':
      return UnetGenerator(1, 1, 7, ngf=64, norm_layer=get_norm_layer(norm_type='batch'), 
                                 use_dropout='False')
    else:
      raise Exception("Invalid generator network name")
  elif type == 'discriminator':
    if name =='dcgan':
      return DCGANDiscriminator()
    elif name == 'patchgan':
      return PatchGANDiscriminator()
    else:
      raise Exception("Invalid discriminator network name")

def get_norm_layer(norm_type='instance'):
    """Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

### Network architectures

class VGG19Generator(nn.Module):
  """
    Based on VGG-19 architecture
    https://arxiv.org/pdf/1704.05838.pdf
  """
  def __init__(self):

    super(VGG19Generator, self).__init__()
    
    n_d_channels = 64

    ### Encoder

    ### 1st CNN downsampling block
    encoder = [nn.Conv2d(1,n_d_channels,(3,3),(1,1),padding=(1,1)),
             nn.ReLU(),
             nn.Conv2d(n_d_channels,n_d_channels,(3,3),(1,1),padding=(1,1)),
             nn.ReLU(inplace=True),
             nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True,)]

    ### 2nd CNN downsampling block
    encoder += [nn.Conv2d(n_d_channels,n_d_channels*2,(3,3),(1,1),padding=(1,1)),
             nn.ReLU(inplace=True),
             nn.Conv2d(n_d_channels*2,n_d_channels*2,(3,3),(1,1),padding=(1,1)),
             nn.ReLU(inplace=True),
             nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)]
    
    ### 3rd CNN downsampling block
    encoder += [nn.Conv2d(n_d_channels*2,n_d_channels*4,(3,3),(1,1),padding=(1,1)),
             nn.ReLU(inplace=True),
             nn.Conv2d(n_d_channels*4,n_d_channels*4,(3,3),(1,1),padding=(1,1)),
             nn.ReLU(inplace=True),
             nn.Conv2d(n_d_channels*4,n_d_channels*4,(3,3),(1,1),padding=(1,1)),
             nn.ReLU(inplace=True),
             nn.Conv2d(n_d_channels*4,n_d_channels*4,(3,3),(1,1),padding=(1,1)),
             nn.ReLU(inplace=True),
             nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)]
    
    ### 4th CNN downsampling block
    encoder += [nn.Conv2d(n_d_channels*4,n_d_channels*8,(3,3),(1,1),padding=(1,1)),
             nn.ReLU(inplace=True),
             nn.Conv2d(n_d_channels*8,n_d_channels*8,(3,3),(1,1),padding=(1,1)),
             nn.ReLU(inplace=True),
             nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)]
    self.encoder = nn.Sequential(*encoder)

    ### FCN
    fcn = [nn.Linear(in_features=32768, out_features=1028, bias=False),
          nn.ReLU(inplace=True),
          nn.Linear(in_features=1028, out_features=32768, bias=False),
          nn.ReLU(inplace=True)]

    self.fcn = nn.Sequential(*fcn)

    ### 1st CNN upsampling Block
    decoder = [nn.MaxUnpool2d(kernel_size=2,stride=2,padding=0),
              nn.ConvTranspose2d(n_d_channels*8,n_d_channels*8,(3,3),(1,1),padding=(1,1)),
              nn.ReLU(inplace=True),
              nn.ConvTranspose2d(n_d_channels*8,n_d_channels*4,(3,3),(1,1),padding=(1,1)),
              nn.ReLU(inplace=True)]

    ### 2nd CNN upsampling Block
    decoder += [nn.MaxUnpool2d(kernel_size=2,stride=2,padding=0),
              nn.ConvTranspose2d(n_d_channels*4,n_d_channels*4,(3,3),(1,1),padding=(1,1)),
              nn.ReLU(inplace=True),
              nn.ConvTranspose2d(n_d_channels*4,n_d_channels*4,(3,3),(1,1),padding=(1,1)),
              nn.ReLU(inplace=True),
              nn.ConvTranspose2d(n_d_channels*4,n_d_channels*4,(3,3),(1,1),padding=(1,1)),
              nn.ReLU(inplace=True),
              nn.ConvTranspose2d(n_d_channels*4,n_d_channels*2,(3,3),(1,1),padding=(1,1)),
              nn.ReLU(inplace=True)]
    
    ### 3rd CNN upsampling Block
    decoder += [nn.MaxUnpool2d(kernel_size=2,stride=2,padding=0,),
              nn.ConvTranspose2d(n_d_channels*2,n_d_channels*2,(3,3),(1,1),padding=(1,1)),
              nn.ReLU(inplace=True),
              nn.ConvTranspose2d(n_d_channels*2,n_d_channels,(3,3),(1,1),padding=(1,1)),
              nn.ReLU(inplace=True)]

    ### 4th CNN upsampling Block
    decoder += [nn.MaxUnpool2d(kernel_size=2,stride=2,padding=0),
              nn.ConvTranspose2d(n_d_channels,n_d_channels,(3,3),(1,1),padding=(1,1)),
              nn.ReLU(inplace=True),
              nn.ConvTranspose2d(n_d_channels,1,(3,3),(1,1),padding=(1,1)),
              nn.ReLU(inplace=True)]
    
    self.decoder = nn.Sequential(*decoder)

  def forward(self, x):
    """Standard forward"""

    ### Forward the encoder
    indices_list = []
    for layer in self.encoder:
      if(isinstance(layer,torch.nn.modules.pooling.MaxPool2d)):
        x, ind = layer(x)
        indices_list.append(ind)
      else:
        x = layer(x)

    x = x.view(-1,32768)
    for layer in self.fcn:
      x = layer(x)
    x = x.view(-1,512,8,8)

    for layer in self.decoder:
      if(isinstance(layer,torch.nn.modules.pooling.MaxUnpool2d)):
        x = layer(x,indices_list.pop())
      else:
        x = layer(x)
    
    return x

class DCGANDiscriminator(nn.Module):
  """
  Based on DC-GAN discriminator architecture
  https://arxiv.org/abs/1409.1556 
  """
  def __init__(self):
    super(DCGANDiscriminator, self).__init__()
    ### Encoder

    n_d_channels = 128
    ### 1st CNN downsampling block
    model = [nn.Conv2d(1,n_d_channels,(5,5),(1,1),padding=(1,1)),
             nn.ReLU(),
             nn.MaxPool2d(kernel_size=2, stride=2)]

    ### 2nd CNN downsampling block
    model += [nn.Conv2d(n_d_channels,n_d_channels*2,(5,5),(1,1),padding=(1,1)),
             nn.ReLU(),
             nn.MaxPool2d(kernel_size=2, stride=2)]
    
    ### 3rd CNN downsampling block
    model += [nn.Conv2d(n_d_channels*2,n_d_channels*4,(5,5),(1,1),padding=(1,1)),
             nn.ReLU(),
             nn.MaxPool2d(kernel_size=2, stride=2)]
    
    ### 4th CNN downsampling block
    model += [nn.Conv2d(n_d_channels*4,n_d_channels*8,(5,5),(1,1),padding=(1,1)),
             nn.ReLU(),
             nn.MaxPool2d(kernel_size=2, stride=2)]
    
    ### FCN
    model += [nn.Linear(in_features=36864, out_features=4096, bias=True),
              nn.ReLU(inplace=True),
              nn.Linear(in_features=4096, out_features=512, bias=True),
              nn.ReLU(inplace=True),
              nn.Linear(in_features=512, out_features=2, bias=True),
              nn.Softmax()]

    self.model = nn.Sequential(*model)


  def forward(self, x):

    switch = True
    for layer in self.model:
      if isinstance(layer,nn.Linear) and switch:
        dim = x.size()
        x = x.view(-1,dim[1]*dim[2]*dim[3])
        switch = False
      x = layer(x)
    return x.view(-1, 1)


### UNet Generator
class UnetGenerator(nn.Module):
    """
      source: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
      Create a Unet-based generator
    """

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """
        source: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py

        Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

### PatchGAN Discriminator
class PatchGANDiscriminator(nn.Module):
    def __init__(self, c=1, n_d_channel=64, sigmoid=True):
        super(PatchGANDiscriminator, self).__init__()
        n_d_channels = 64
        modules = [
            # input is c x 64 x 64
            nn.Conv2d(c, n_d_channels, kernel_size=(4, 4), stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2),
            # state size. (n_d_channels) x 32 x 32
            nn.Conv2d(n_d_channels, n_d_channels * 2, kernel_size=(4, 4), stride=2, padding=1, bias=False),
            nn.BatchNorm2d(n_d_channels * 2),
            nn.LeakyReLU(0.2),
            # state size. (n_d_channels*2) x 16 x 16
            nn.Conv2d(n_d_channels * 2, n_d_channels * 4, kernel_size=(4, 4), stride=2, padding=1, bias=False),
            nn.BatchNorm2d(n_d_channels * 4),
            nn.LeakyReLU(0.2),
            # state size. (n_d_channels*4) x 8 x 8
            nn.Conv2d(n_d_channels * 4, n_d_channels * 8, kernel_size=(4, 4), stride=2, padding=1, bias=False),
            nn.BatchNorm2d(n_d_channels * 8),
            nn.LeakyReLU(0.2),
            # state size. (n_d_channels*8) x 4 x 4
            nn.Conv2d(n_d_channels * 8, 1, kernel_size=(4, 4), stride=1, padding=0, bias=False),
            Flatten(),
            nn.Linear(25,1)
        ]
        if sigmoid:
          modules.append(nn.Sigmoid())

        self.model = nn.Sequential(*modules)

    def forward(self, x):
        output = self.model(x)
        return output.view(-1, 1)