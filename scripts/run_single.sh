#!/bin/bash
data_dir=data/processed_data

sup_cut=0.05
unsup_cut=1.0
batch_size=2
unsup_ratio=3
seed=44
early_stop_steps=20000

model_dir="ckpt/model_sup_${sup_cut}_unsup_${unsup_cut}_seed_${seed}"

python main.py \
  --do_eval_along_training=True \
  --do_predict=False \
  --sup_cut=${sup_cut[i]} \
  --unsup_cut=${unsup_cut} \
  --unsup_ratio=${unsup_ratio} \
  --shuffle_seed=${seed} \
  --train_batch_size=${batch_size} \
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