PYTHONPATH=./ python src/experiments/validate_sae/attribute_prediction/evaluate_sae_prediction.py \
    --device cuda:1 \
    --sae_path ./out/checkpoints/openai_l14_32K_base/clip-vit-large-patch14_-2_resid_32768.pt \
