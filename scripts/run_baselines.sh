#!/bin/bash
sup_cuts="0.01 0.05 0.1 1.0"
data_dir=data/processed_data

for experiment_number in {1..3}; do
  for sup_cut in $sup_cuts; do
    model_dir="ckpt/new_baseline_${sup_cut}_supervised_${experiment_number}"

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
  done
done