#!/bin/bash

#SBATCH --time=06:00:00
#SBATCH --output=%x.out
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G


cd ~/federated/

if [ ! -d venv ]; then
  module load python/3.8
  virtualenv --no-download venv
  source venv/bin/activate
  pip install --no-index tensorflow_gpu
else
  source venv/bin/activate
fi


export ADNI_ROOT=./dataset
export RESULTS_ROOT=/scratch/aminesi/federated/results
export CONFIG_PATH="./configs/config-$CONF_ID.json"

python ./test.py
deactivate
