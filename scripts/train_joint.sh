#!/bin/zsh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=96:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=research

module load nvidia/cuda/11.3
python setup.py build_ext --inplace
python train_joint.py -c configs/joint/$1.yaml -n $2 -g $3

# EXAMPLES:
# sbatch --gres=gpu:1 --nodelist=euler20 train_joint.sh train_ch1_128 joint_ch1_128 0
# sbatch --gres=gpu:2 --nodelist=euler19 train_joint.sh train_ch1_256 joint_ch1_256 0,1

# NOTE: set $3 to 0 for 1 GPU, 0,1 for 2 GPUs, 0,1,2,3 for 4 GPUs, and so on.