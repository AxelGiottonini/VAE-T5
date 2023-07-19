#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:lmol
#SBATCH --time=20:00:00

CUDA_VISIBLE_DEVICES=0 python ./train.py \
    --model_name "test" \
    --model_version "0.1" \
    --training_set "./data/_test.csv" \
    --validation_set "./data/_val.csv" \
    --max_length 512 \
    --mask \
    --mask_rate 0.15 \
    --frag_coef_a 3 \
    --frag_coef_b 2 \
    --n_epochs 1 \
    --global_batch_size 512 \
    --local_batch_size 32 \
    --vae_dims_hidden "512, 256" \
    --vae_dim_latent "128" \
    --vae_dropout_p "0.2" \
    --vae_dim_discriminator 32 \
    --f_model 1 \
    --f_recon 10 \
    --f_kld 0 \
    --f_adv 5 \
    --f_var 0 \
    --teacher_forcing 0 