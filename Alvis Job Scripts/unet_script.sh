#!/usr/bin/env bash
#SBATCH -A SNIC2021-7-47 -p alvis
#SBATCH -N 1 --gpus-per-node=V100:1  # We're launching 2 nodes with 4 Nvidia T4 GPUs each
python3 unet_docs.py
