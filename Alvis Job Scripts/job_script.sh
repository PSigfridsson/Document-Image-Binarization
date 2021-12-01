#!/usr/bin/env bash
#SBATCH -A SNIC2021-7-47 -p alvis
#SBATCH -N 1 --gpus-per-node=V100:1  # We're launching 1 nodes with 4 Nvidia T4 GPUs each
#SBATCH -t 0-00:10:00
#SBATCH -o jobscript_output_2
echo "Hello cluster computing world!"
sleep 60

#Here you should typically call your GPU-hungry application
