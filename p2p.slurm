#!/bin/bash
#SBATCH --job-name="P2P Test"
#SBATCH --output="p2p.out"
#SBATCH --partition=A100q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --account=bche-delta-gpu
#SBATCH -t 24:00:00

module reset
module load cuda

echo -e "job $SLURM_JOBID started on `hostname`\n\n"
srun python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA >> p2p.out