#!/bin/bash
sup_cut=(0.01 0.05 0.1)
unsup_cut=(0.99 0.95 0.9)
data_dir=data/processed_data

for experiment_nbr in {4..4}; do
  for i in {1..1}; do
    model_dir="ckpt/${sup_cut[i]}_${unsup_cut[i]}_uda_${experiment_nbr}"

    python main.py \
      --do_eval_along_training=True \
      --do_predict=False \
      --sup_cut=${sup_cut[i]} \
      --unsup_cut=${unsup_cut[i]} \
      --unsup_ratio=2 \
      --shuffle_seed=42 \
      --train_batch_size=1 \
      --train_steps=100000 \
      --max_save=1 \
      --data_dir=${data_dir} \
      --model_dir=${model_dir} \
      --unsup_coeff=1 \
      --tsa= \
      --early_stop_steps=10000 \
      --unsup_crop=True \
      --exp_lr_decay=True

    python main.py \
      --do_eval_along_training=False \
      --do_predict=True \
      --data_dir=${data_dir} \
      --eval_batch_size=16 \
      --model_dir=${model_dir} \
      --pred_dataset=val
  done
done
