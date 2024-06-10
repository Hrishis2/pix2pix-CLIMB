#!/bin/bash

# Base/Directory Flags
dataroot="./datasets/facades"           # Path to the directory where the dataset for train/test/val is stored

name="facades_pix2pix"                  # Name of the experiment, used to store samples and models

gpu_ids="0"                       # Individual GPU ids to utilize, so if using 4 GPUs gpu_ids="0,1,2,3". 
                                        # If gpu_ids="-1" then CPU mode is used

checkpoints_dir="./checkpoints"         # Path to directory to store checkpoints of model after each epoch


# Model Flags
model="pix2pix"                         # cycle_gan | pix2pix | test | colorization, specifies which of these models to use

input_nc=3                              # Input channels, 3 for RGB, 1 for grayscale

output_nc=3                             # Output channels, 3 for RGB, 1 for grayscale

ngf=64                                  # Number of gen filters at the last convolutional layer

ndf=64                                  # Number of discriminatory filters at the last convolutional layer

netD="basic"                            # basic | n_layers | pixel, where basic is 70x70 Patch GAN. 
                                        # Pixel is 1x1 Patch GAN, n_layers allows you to specify

netG="unet_256"                         # resnet_9blocks | resnet_6blocks | unet_256 | unet_128, specifies generator architecture

n_layers_D=3                            # Only used if netD=n_layers

norm="batch"                            # instance | batch | none, the type of normalization

init_type="normal"                      # normal | xavier | kaiming | orthogonal, 
                                        # type of network initialization, such as setting initial weights

init_gain=0.02                          # scaling factor for normal, xavier, or orthogonal

# Dataset Flags
dataset_mode="aligned"                  # Unaligned = unpaired, such as for CycleGAN. Aligned = paired, such as pix2pix. 
                                        # Single = only input needed, such as colorization.

direction="BtoA"                        # AtoB | BtoA, based on dataset, which way to train the model. 
                                        # So if B = horse, and A = zebra, then BtoA will make zebras from horses.

num_threads=4                           # Number of threads used to load in data

batch_size=1                            # The batch size for loading input data, should be greater than 1 for multi-GPU to have a positive effect

load_size=286                           # Scales input images up or down to this size

crop_size=256                           # Then crops a random square of this size from the input

# max_dataset_size="float("inf")"         # Maximum samples allowed per dataset
preprocess="resize_and_crop"            # resize_and_crop | crop | scale_width | scale_width_and_crop | none. Scaling and cropping of images at load time

display_winsize=256                     # display window size for HTML

# Additional Flags
epoch="latest"                          # Which epoch to load, set to latest to use the latest cached model

load_iter=0                             # Which iteration to load, if load_iter > 0, the code will load models by iter_[load_iter]; 
                                        # otherwise, the code will load models by [epoch]
                                        
# suffix=""                              # Customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}

sbatch <<EOT
#!/bin/bash
#SBATCH --job-name="P2P Train"          # 1 - The name of the job as shown on Delta queue
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
srun python train.py --dataroot ${dataroot} --name ${name} --gpu_ids ${gpu_ids} --checkpoints_dir ${checkpoints_dir} --model ${model} --input_nc ${input_nc} --output_nc ${output_nc} --ngf ${ngf} --ndf ${ndf} --netD ${netD} --netG ${netG} --n_layers_D ${n_layers_D} --norm ${norm} --init_type ${init_type} --init_gain ${init_gain} ${no_dropout:+--no_dropout} --dataset_mode ${dataset_mode} --direction ${direction} ${serial_batches:+--serial_batches} --num_threads ${num_threads} --batch_size ${batch_size} --load_size ${load_size} --crop_size ${crop_size} --preprocess ${preprocess} ${no_flip:+--no_flip} --display_winsize ${display_winsize} --epoch ${epoch} --load_iter ${load_iter} ${verbose:+--verbose} >> p2p.out

hostname

exit 0
EOT