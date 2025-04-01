#!/bin/bash
NUM_GPUS=4
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Specify which GPUs to use

# For training
torchrun --nproc_per_node=$NUM_GPUS train_xcoax.py

# For finetuning (uncomment to use)
# torchrun --nproc_per_node=$NUM_GPUS finetune_xcoax.py --model-path checkpoints/model.pt 