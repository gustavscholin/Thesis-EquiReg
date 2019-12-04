#!/bin/bash
sup_cut=$1
model_dir="ckpt/${sup_cut}_supervised"
data_dir=data/processed_data

python main.py \
  --do_eval_along_training=True \
  --do_predict=False \
  --sup_cut=${sup_cut} \
  --unsup_cut=0. \
  --unsup_ratio=0 \
  --shuffle_seed=42 \
  --train_batch_size=4 \
  --train_steps=100000 \
  --max_save=1 \
  --data_dir=${data_dir} \
  --model_dir=${model_dir}

python main.py \
  --do_eval_along_training=False \
  --do_predict=True \
  --data_dir=${data_dir} \
  --eval_batch_size=16 \
  --model_dir=${model_dir} \
  --pred_dataset=val