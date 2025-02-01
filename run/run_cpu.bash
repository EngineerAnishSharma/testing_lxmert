#!/bin/bash

# The name of this experiment.
name=$1

# Save logs and models under snap/vqa; make backup.
output=snap/vqa/$name
mkdir -p $output/src
cp -r src/* $output/src/
cp $0 $output/run_cpu.bash

# Run on CPU (no CUDA_VISIBLE_DEVICES)
# Important: Adjust paths to your data and pretrained model (if using)
PYTHONPATH=$PYTHONPATH:./src \
    python src/tasks/vqa.py \
    --train data/vqa_bias_mitigated/preprocessed_vqa_bias_mitigated_train.json \
    --valid data/vqa_bias_mitigated/preprocessed_vqa_bias_mitigated_val.json \
    --llayers 9 --xlayers 5 --rlayers 5 \
    --loadLXMERTQA snap/pretrained/model \
    --batchSize 4 --optim bert --lr 5e-5 --epochs 2 \
    --tqdm --output $output ${@:2}