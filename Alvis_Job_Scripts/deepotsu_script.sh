#!/usr/bin/env bash
#SBATCH -A SNIC2021-7-47 -p alvis
#SBATCH -N 1 --gpus-per-node=V100:1  # We're launching 2 nodes with 4 Nvidia T4 GPUs each
#SBATCH -t 0-24:00:00


cd ../new_deepOtsu
singularity exec ../../env.sif python3 trainNetwork.py -gpu -se 10000 -ep 1 -ds Cuper nabuco irish ICDAR_1 ICDAR_5 ICDAR_10 ICDAR_21 ICDAR_41 ICDAR_28 ICDAR_33
