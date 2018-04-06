#!/bin/sh

echo "Set up training configuration"

export OUTPUT_POSTFIX="run_1"
export USE_WHOLE_DATASET="False"
export OUTPUT_FOLDER="output"
export DATA_PATH="data/encoded"
export ENCODING_NAME="standard"
export EPOCHS_COUNT=1000
export BATCH_SIZE=100
export LEARNING_RATE=0.0001
export DROPOUT_RATE=0.25
export RUN_MODE="train"
export RESTORE_CHECKPOINT_PATH="./output/checkpoints/standard_run_1/model_epoch49.ckpt"

. ./scripts/run_train.sh