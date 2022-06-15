# finsim2022
Codebase to reproduce my workflow for FinSim 2022

## Data download and preparation

## Finetuning DeBERTa
To reproduce our finetuning, run the command:
```commandline
HF_MODEL="microsoft/deberta-v3-large"
MODEL_NAME=$(echo $HF_MODEL | rev | cut -d/ -f1 | rev)

python train.py \
  --model_name_or_path $HF_MODEL \
  --do_train \
  --do_eval \
  --train_file data/processed/train.csv \
  --validation_file data/processed/dev.csv \
  --max_seq_length 64 \
  --per_device_train_batch_size 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 2 \
  --output_dir checkpoints/${MODEL_NAME}-finetuned-finsim4-esg/ \
  --cache_dir .cache/ \
  --overwrite_output_dir \
  --pad_to_max_length \
  --seed 2022 \
  --data_seed 2022 \
  --fp16 \
  --load_best_model_at_end \
  --metric_for_best_model eval_accuracy \
  --evaluation_strategy steps \
  --group_by_length \
  --label_smoothing_factor 0.1 \
  --early_stopping_patience 2 \
  --early_stopping_threshold 0.001
```

# Inference with finetuned model

# Prepare submission
