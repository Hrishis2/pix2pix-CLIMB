#!/bin/bash
dataroot="./datasets/facades"
name="facades_pix2pix"
model="pix2pix"
direction="BtoA"
sbatch <<EOT
#!/bin/bash
#SBATCH --job-name="P2P Test"           # 1 - The name of the job as shown on Delta queue
#SBATCH --output="p2p.out"              # 2 - The output file of any print statements during testing, logs progress and reflect errors
#SBATCH --partition=A100q,A40q          # 3 - The type of nodes chosen for the job. Note that commas choose one of the options
#SBATCH --nodes=1                       # 4 - The number of nodes needed for the job, should remain 1 unless training multiple at once
#SBATCH --mem=208G                      # 5 - The number of memory needed for the node, defaults to 1G but may need more
#SBATCH --ntasks-per-node=1             # 6 - Number of tasks per node, should be 1
#SBATCH --cpus-per-task=64              # 7 - The number of CPU cores to use, usually 16 times number of GPUs
#SBATCH --gpus-per-node=4               # 8 - How many GPUs per node (and in turn, per task). Needs to reflect command line args
#SBATCH --gpu-bind=closest              # 9 - GPU binding technique, don't need to change
#SBATCH --account=bche-delta-gpu        # 10 - Account name
#SBATCH --exclusive                     # 11 - Exclusively take up an entire node
#SBATCH --no-requeue                    # 12 - Do not requeue in event of a node failure
#SBATCH -t 24:00:00                     # 13 - Amount of time allocated for a job, after which the job is timed out



echo -e "job $SLURM_JOBID started on `hostname`\n\n"
srun python test.py --dataroot ${dataroot} --name ${name} --model ${model} --direction ${direction} --gpu_ids 0,1,2,3 --batch_size 16 >> p2p.out

hostname

exit 0
EOT