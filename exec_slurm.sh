#! /bin/bash

### *usage*
### sbatch train.sh
### common partition: --partition=gpu_1080-ti

#SBATCH --gres=gpu:1
#SBATCH -o slurm.out
#SBATCH -N1

source activate py3
srun python -u train.py --experiments minimaxgan_rmse wgan_rmse minimaxgan_l1 wgan_l1 wgan_perceptual_style_faceparsing -ep 200 -b 64 --saveevery 20 --evalevery 5

### debug
### srun python -u train.py --experiments minimaxgan_rmse wgan_rmse minimaxgan_l1 wgan_l1 wgan_perceptual_style_faceparsing -ep 1 -b 64 --saveevery 1 --evalevery 1 --debug true