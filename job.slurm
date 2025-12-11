#!/bin/bash
#SBATCH --nodes=1                    # on demande un noeud
#SBATCH --ntasks-per-node=1          # avec une tache par noeud (= nombre de GPU ici)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --account hwy@v100
#SBATCH --time=19:00:00
#SBATCH --qos=qos_gpu-t3

## load Pytorch module
module purge
module load miniforge/24.11.3
module load git
conda activate /lustre/fswork/projects/rech/hir/ukz42ac/.conda/envs/dreamer

# export PATH=~/.local/bin:$PATH
#export MKL_SERVICE_FORCE_INTEL=1

#export OMP_NUM_THREADS=1
#export MKL_NUM_THREADS=1 
#conda activate dnadna

# dnadna preprocess "/gpfsdswork/projects/rech/hwy/uyh28br/SPIDNA/SPIBAC_models_preprocessing_config.yml"
# export PYTHONPATH=/lustre/fswork/projects/rech/hwy/ukz42ac/src/dnadna:$PYTHONPATH

python train.py --conf $1 --seed $2
