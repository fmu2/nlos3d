#!/bin/zsh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=96:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=research

module load nvidia/cuda/11.3
python setup.py build_ext --inplace
python train_nr.py -c configs/nr/$1.yaml -n $2 -g $3

# sbatch --gres=gpu:2 --nodelist=euler19 train_nr.sh $1 $2 $3
# NOTE: set $3 to 0 for 1 GPU, 0,1 for 2 GPUs, 0,1,2,3 for 4 GPUs, and so on.