PYTHONPATH=./ python src/experiments/validate_sae/attribute_prediction/evaluate_vlm_prediction.py \
    --device cuda:1 \
    --model_name llava_next \
    --dataset_name all \
    --split test \

