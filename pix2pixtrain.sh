#!/bin/bash

# Base/Directory Flags
dataroot="./datasets/facades"           #1 - Path to the directory where the dataset for train/test/val is stored

name="facades_pix2pix"                  #2 - Name of the experiment, used to store samples and models

gpu_ids="0"                             #3 - Individual GPU ids to utilize, so if using 4 GPUs gpu_ids="0,1,2,3". 
                                        # If gpu_ids="-1" then CPU mode is used

checkpoints_dir="./checkpoints"         #4 - Path to directory to store checkpoints of model after each epoch


# Model Flags
model="pix2pix"                         #5 - cycle_gan | pix2pix | test | colorization, specifies which of these models to use

input_nc=3                              #6 - Input channels, 3 for RGB, 1 for grayscale

output_nc=3                             #7 - Output channels, 3 for RGB, 1 for grayscale

ngf=64                                  #8 - Number of gen filters at the last convolutional layer

ndf=64                                  #9 - Number of discriminatory filters at the last convolutional layer

netD="basic"                            #10 - basic | n_layers | pixel, where basic is 70x70 Patch GAN. 
                                        # Pixel is 1x1 Patch GAN, n_layers allows you to specify

netG="unet_256"                         #11 - resnet_9blocks | resnet_6blocks | unet_256 | unet_128, specifies generator architecture

n_layers_D=3                            #12 - Only used if netD=n_layers

norm="batch"                            #13 - instance | batch | none, the type of normalization

init_type="normal"                      #14 - normal | xavier | kaiming | orthogonal, 
                                        # type of network initialization, such as setting initial weights

init_gain=0.02                          #15 - scaling factor for normal, xavier, or orthogonal

no_dropout=                             #16 - If set to true, will disable dropout. Dropout minimizes overfitting by dropping out nodes, else set blank

# Dataset Flags
dataset_mode="aligned"                  #17 - Unaligned = unpaired, such as for CycleGAN. Aligned = paired, such as pix2pix. 
                                        # Single = only input needed, such as colorization.

serial_batches=                         #18 - If set to true, batches will be chosen in sequential order i.e. 1-50, then 50-100, etc, else set blank

direction="BtoA"                        #19 - AtoB | BtoA, based on dataset, which way to train the model. 
                                        # So if B = horse, and A = zebra, then BtoA will make zebras from horses.

num_threads=4                           #20 - Number of threads used to load in data

batch_size=1                            #21 - The batch size for loading input data, should be greater than 1 for multi-GPU to have a positive effect

load_size=286                           #22 - Scales input images up or down to this size

crop_size=256                           #23 - Then crops a random square of this size from the input

# max_dataset_size="float("inf")"       #24 - Maximum samples allowed per dataset

preprocess="resize_and_crop"            #25 - resize_and_crop | crop | scale_width | scale_width_and_crop | none. Scaling and cropping of images at load time

no_flip=                                #26 - Disables flipping of images in order to diversity data when set to true, else set blank

display_winsize=256                     #27 - display window size for HTML

# Additional Flags
epoch="latest"                          #28 - Which epoch to load, set to latest to use the latest cached model

load_iter=0                             #29 - Which iteration to load, if load_iter > 0, the code will load models by iter_[load_iter]; 
                                        # otherwise, the code will load models by [epoch]

verbose=                                #30 - If set to true then more debugging information is printed in p2p.out, else set blank

# suffix=""                             #31 - Customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}

# Network saving and loading parameters
save_latest_freq=5000                   #32 - Frequency of saving the latest results

save_epoch_freq=5                       #33 - Frequency of saving checkpoints at the end of epochs

save_by_iter=                           #34 - If true, the model will be saved at each iteration. Else, leave blank

continue_train=false                    #35 - If true, will continue training by loading the latest model. Else, leave blank

epoch_count=1                           #36 - The starting epoch count

phase="train"                           #37 - Train, val, test, etc.

# Training parameters
n_epochs=100                            #38 - Number of epochs with the initial learning rate

n_epochs_decay=100                      #39 - Number of epochs to linearly decay learning rate to zero

beta1=0.5                               #40 - Momentum term of Adam optimizer

lr=0.0002                               #41 - Initial learning rate for Adam

gan_mode="lsgan"                        #42 - vanilla | lsgan | wgangp, the type of GAN objective

pool_size=50                            #43 - Size of image buffer that stores previously generated images

lr_policy="linear"                      #44 - linear | step | plateau | cosine, learning rate policy

lr_decay_iters=50                       #45 - Multiply by a gamma every lr_decay_iters iterations


sbatch <<EOT
#!/bin/bash
#SBATCH --job-name="P2P Train"          #46 - The name of the job as shown on Delta queue
#SBATCH --output="p2p.out"              #47 - The output file of any print statements during testing, logs progress and reflect errors
#SBATCH --partition=A100q,A40q          #48 - The type of nodes chosen for the job. Note that commas choose one of the options
#SBATCH --nodes=1                       #49 - The number of nodes needed for the job, should remain 1 unless training multiple at once
#SBATCH --mem=208G                      #50 - The number of memory needed for the node, defaults to 1G but may need more
#SBATCH --ntasks-per-node=1             #51 - Number of tasks per node, should be 1
#SBATCH --cpus-per-task=64              #52 - The number of CPU cores to use, usually 16 times number of GPUs
#SBATCH --gpus-per-node=4               #53 - How many GPUs per node (and in turn, per task). Needs to reflect command line args
#SBATCH --gpu-bind=closest              #54 - GPU binding technique, don't need to change
#SBATCH --account=bche-delta-gpu        #55 - Account name
#SBATCH --exclusive                     #56 - Exclusively take up an entire node
#SBATCH --no-requeue                    #57 - Do not requeue in event of a node failure
#SBATCH -t 24:00:00                     #58 - Amount of time allocated for a job, after which the job is timed out



echo -e "job $SLURM_JOBID started on `hostname`\n\n"
srun python train.py --dataroot ${dataroot} --name ${name} --gpu_ids ${gpu_ids} --checkpoints_dir ${checkpoints_dir} --model ${model} --input_nc ${input_nc} --output_nc ${output_nc} --ngf ${ngf} --ndf ${ndf} --netD ${netD} --netG ${netG} --n_layers_D ${n_layers_D} --norm ${norm} --init_type ${init_type} --init_gain ${init_gain} ${no_dropout:+--no_dropout} --dataset_mode ${dataset_mode} --direction ${direction} ${serial_batches:+--serial_batches} --num_threads ${num_threads} --batch_size ${batch_size} --load_size ${load_size} --crop_size ${crop_size} --preprocess ${preprocess} ${no_flip:+--no_flip} --display_winsize ${display_winsize} --epoch ${epoch} --load_iter ${load_iter} ${verbose:+--verbose} --save_latest_freq ${save_latest_freq} --save_epoch_freq ${save_epoch_freq} ${save_by_iter:+--save_by_iter} ${continue_train:+--continue_train} --epoch_count ${epoch_count} --phase ${phase} --n_epochs ${n_epochs} --n_epochs_decay ${n_epochs_decay} --beta1 ${beta1} --lr ${lr} --gan_mode ${gan_mode} --pool_size ${pool_size} --lr_policy ${lr_policy} --lr_decay_iters ${lr_decay_iters} >> p2p.out

hostname

exit 0
EOT