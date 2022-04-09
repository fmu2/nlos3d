#!/bin/zsh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=research

module load nvidia/cuda/11.3
python setup.py build_ext --inplace
python test_nr.py -ckpt $1 -c configs/nr/$2.yaml

# NOTE: set $1 to the name of checkpoint folder