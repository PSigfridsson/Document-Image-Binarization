#!/usr/bin/env bash
#SBATCH -A SNIC2021-7-47 -p alvis
#SBATCH -N 1 --gpus-per-node=T4:1  # We're launching 2 nodes with 4 Nvidia T4 GPUs each
#SBATCH -t 0-24:00:00


cd ../u-net
singularity exec ../../env.sif python3 unet_docs.py -gpu -pred -name 1_9999_1e-05_20_0.1_ICDAR*_Cuper_nabuco
 
