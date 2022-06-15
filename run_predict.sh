#!/bin/bash

HF_MODEL="checkpoints/deberta-v3-large-finetuned-finsim4-esg"
MODEL_NAME=$(echo $HF_MODEL | rev | cut -d/ -f1 | rev)

python predict.py \
  --model_name_or_path $HF_MODEL \
  --test_file data/processed/test.csv \
  --do_predict \
  --max_seq_length 128 \
  --per_device_eval_batch_size 256 \
  --output_dir predictions/${MODEL_NAME}/ \
  --cache_dir .cache/ \
  --overwrite_output_dir \
  --pad_to_max_length \
  --seed 2022 \
  --data_seed 2022 \
  --fp16
