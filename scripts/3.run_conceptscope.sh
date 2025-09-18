PYTHONPATH=./ python src/conceptscope/main.py \
    --device cuda:6 \
    --sae_path ./out/checkpoints/openai_l14_32K_base/clip-vit-large-patch14_-2_resid_32768.pt \
    --dataset_name imagenet \
    --split train \
    --batch_size 128 \
    --num_samples 128 \
    --target_threshold 1.0 \
    --clip_model_name openai/clip-vit-large-patch14 \
 
