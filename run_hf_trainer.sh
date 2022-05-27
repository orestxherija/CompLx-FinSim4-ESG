#!/bin/bash
HF_MODEL="espejelomar/beto-base-cased"
MODEL_NAME=$(echo $HF_MODEL | rev | cut -d/ -f1 | rev)
TIMESTAMP=$(date +"%Y%m%d%H%M%S")

python hf_train.py \
  --model_name_or_path $HF_MODEL \
  --do_train \
  --do_eval \
  --train_file train_new.csv \
  --validation_file dev_new.csv \
  --max_seq_length 128 \
  --per_device_train_batch_size 64 \
  --learning_rate 2e-5 \
  --num_train_epochs 25 \
  --output_dir checkpoints/${MODEL_NAME}-${TIMESTAMP}/ \
  --cache_dir .cache/ \
  --overwrite_output_dir \
  --pad_to_max_length \
  --seed 2022 \
  --data_seed 2022 \
  --fp16 \
  --load_best_model_at_end \
  --metric_for_best_model eval_f1 \
  --evaluation_strategy steps \
  --group_by_length \
  --label_smoothing_factor 0.1 \
  --early_stopping_patience 2 \
  --early_stopping_threshold 0.001
