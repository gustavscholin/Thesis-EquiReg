#!/bin/bash
sup_cut=$1
unsup_cut=$2
model_dir="ckpt/${sup_cut}_${unsup_cut}_uda"
data_dir=data/processed_data

python main.py \
  --do_eval_along_training=True \
  --do_predict=False \
  --sup_cut=${sup_cut} \
  --unsup_cut=${unsup_cut} \
  --unsup_ratio=2 \
  --shuffle_seed=42 \
  --train_batch_size=1 \
  --train_steps=70000 \
  --max_save=1 \
  --data_dir=${data_dir} \
  --model_dir=${model_dir} \
  --unsup_coeff=1 \
  --tsa=log_schedule \
  --early_stop_steps=-1 \
  --unsup_crop=True

python main.py \
  --do_eval_along_training=False \
  --do_predict=True \
  --data_dir=${data_dir} \
  --eval_batch_size=16 \
  --model_dir=${model_dir} \
  --pred_dataset=val
