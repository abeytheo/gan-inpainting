
# Image Inpainting Using GAN

Given ground truth images and their corresponding binary masks, the goal of an image inpainting GAN is to fill the masked region with appropriate pixels, forming semantically correct images. This repository specifically contains my experiment pipeline for my master thesis. The pipeline structure has been designed to support multiple modular experiments.

The experiments can be executed in a slurm cluster via the following command:
```
sbatch exec_slurm.sh
```