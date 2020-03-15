#!/bin/bash
data_dir=data/processed_data

sup_cut=0.05
experiment_nbr=1
seed=44
early_stop_steps=20000

model_dir="/mnt/storage/data/thesis-uda/ckpt/equireg/equireg_${sup_cut}_${experiment_nbr}_seed_${seed}"

python main.py \
  --do_eval_along_training=True \
  --do_predict=False \
  --sup_cut=${sup_cut[i]} \
  --unsup_cut=1.0 \
  --unsup_ratio=3 \
  --shuffle_seed=${seed} \
  --train_batch_size=2 \
  --eval_batch_size=14 \
  --train_steps=200000 \
  --max_save=1 \
  --data_dir=${data_dir} \
  --model_dir=${model_dir} \
  --unsup_coeff=1 \
  --tsa= \
  --early_stop_steps=${early_stop_steps} \
  --min_step=0 \
  --unsup_crop=True \
  --exp_lr_decay=False

python main.py \
  --do_eval_along_training=False \
  --do_predict=True \
  --data_dir=${data_dir} \
  --eval_batch_size=16 \
  --model_dir=${model_dir} \
  --pred_dataset=val

python main.py \
  --do_eval_along_training=False \
  --do_predict=True \
  --data_dir=${data_dir} \
  --eval_batch_size=16 \
  --model_dir=${model_dir} \
  --pred_dataset=test