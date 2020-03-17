
# Image Inpainting Using GAN

Given a ground truth image and a binary mask, the goal of an image inpainting GAN is to learn how to restore the masked image to a semantically correct image. This repository specifically contains my experiment codes for my master thesis.

Usage:
```bash
python train.py -t <Experiment title>
```

```bash
Training configurations

optional arguments:
  -h, --help            show this help message and exit
  -t TITLE, --title TITLE
                        Experiment title
  -e [NUMEPOCH], --numepoch [NUMEPOCH]
                        Number of training epoch
  -b [BATCHSIZE], --batchsize [BATCHSIZE]
                        Batch size in one training epoch
  -g [{unet,vgg19}], --generator [{unet,vgg19}]
                        Generator network name
  -d [{patchgan,dcgan}], --discriminator [{patchgan,dcgan}]
                        Discriminator network name
  --imagedim [IMAGEDIM]
                        Image dimension
  --saveevery [SAVEEVERY]
                        Save network every N epoch(s)
  --updatediscevery [UPDATEDISCEVERY]
                        Backprop discriminator every N epoch(s)
  --lambda1 [LAMBDA1]   Hyperparameter lambda 1
  --lambda2 [LAMBDA2]   Hyperparameter lambda 2
```