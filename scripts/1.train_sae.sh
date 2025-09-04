PYTHONPATH=./ python src/sae_training/train_sae.py \
    --device cuda:0 \
    --block_layer -2 \
    --use_ghost_grads \
    --seed 1 \
    --n_checkpoints 1 \
    --total_training_tokens 5000000 \
    --log_to_wandb \
    --model_name openai/clip-vit-large-patch14 \
    --expansion_factor 32 \
    --clip_dim 1024 \


