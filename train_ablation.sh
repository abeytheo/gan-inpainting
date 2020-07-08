#! /bin/bash

#SBATCH --gres=gpu:1
#SBATCH -c 8 --mem 16G
#SBATCH -o slurm_ablation.out
#SBATCH -N1

source activate py3
srun python -u train.py --experiments ablation_wgan_adversarial ablation_wgan_parsing ablation_wgan_perceptual ablation_wgan_style ablation_wgan_tv -ep 300 -b 64 --saveevery 20 --evalevery 5
srun python -u train.py --experiments curriculum1_wgan_l2_percept curriculum1_wgan_l2_percept_tv curriculum1_wgan_l2_style curriculum1_wgan_l2_style_tv curriculum1_wgan_l2_style_percept curriculum1_wgan_l2_style_percept_tv curriculum1_wgan_l2_faceparsing -ep 300 -b 64 --saveevery 20 --evalevery 5

### debug
# srun python -u train.py --experiments ablation_wgan_adversarial ablation_wgan_parsing ablation_wgan_perceptual ablation_wgan_style ablation_wgan_tv -ep 1 -b 64 --saveevery 1 --evalevery 1 --debug true
# srun python -u train.py --experiments curriculum1_wgan_l2_percept curriculum1_wgan_l2_percept_tv curriculum1_wgan_l2_style curriculum1_wgan_l2_style_tv curriculum1_wgan_l2_style_percept curriculum1_wgan_l2_style_percept_tv curriculum1_wgan_l2_faceparsing -ep 1 -b 64 --saveevery 1 --evalevery 1 --debug true