#!/bin/bash
MODEL_NAME='resnet50'
# MODEL_NAME='resnet101'
# MODEL_NAME='resnet152'

# SAVE_MODEL_PATH="./results/ssd.pth"
# SAVE_MODEL_PATH="./results/fcos.pth"
# SAVE_MODEL_PATH="./results/retina.pth"
# SAVE_MODEL_PATH="./results/faster.pth"
# SAVE_MODEL_PATH="./results/cascade.pth"
SAVE_MODEL_PATH="./results/sparse.pth"

# LOG_DIR="./logs/ssd"
# LOG_DIR="./logs/fcos"
# LOG_DIR="./logs/retina"
# LOG_DIR="./logs/faster"
# LOG_DIR="./logs/cascade"
LOG_DIR="./logs/sparse"

NUM_EPOCHS=50


python train.py \
    --model_name $MODEL_NAME \
    --num_epochs $NUM_EPOCHS \
    --save_model_path $SAVE_MODEL_PATH \
    --log_dir $LOG_DIR \
    AdamW
