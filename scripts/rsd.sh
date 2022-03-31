#!/bin/zsh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=anything

module load nvidia/cuda/11.3
python setup.py build_ext --inplace
python rsd.py -c configs/rsd/$1.yaml -n $2