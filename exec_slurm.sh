#! /bin/bash

### *usage*
### sbatch train.sh
### common partition: --partition=gpu_1080-ti

#SBATCH --gres=gpu:1
#SBATCH -o slurm.out
#SBATCH -N1

source activate py3
srun python -u train.py -t experiment1