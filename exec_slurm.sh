#! /bin/bash

### *usage*
### sbatch train.sh
### common partition: --partition=gpu_1080-ti

#SBATCH --gres=gpu:1
#SBATCH -o slurm.out
#SBATCH -N1

source activate py3
srun python -u train.py --experiments minimaxgan_rmse wgan_rmse -ep 400 -b 64 --saveevery 20 --evalevery 10