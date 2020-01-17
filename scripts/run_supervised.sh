#!/bin/bash
sup_cut=0.01
experiment_number=1
seed=43
save_steps=125
early_stop_steps=2000
model_dir="ckpt/baseline_${sup_cut}_${experiment_number}_seed_${seed}"
data_dir=data/processed_data

python main.py \
  --do_eval_along_training=True \
  --do_predict=False \
  --sup_cut=${sup_cut} \
  --unsup_cut=0. \
  --unsup_ratio=0 \
  --shuffle_seed=${seed} \
  --train_batch_size=4 \
  --train_steps=100000 \
  --save_steps=${save_steps} \
  --max_save=1 \
  --data_dir=${data_dir} \
  --model_dir=${model_dir} \
  --early_stop_steps=${early_stop_steps}  \
  --min_step=0 \
  --exp_lr_decay=False

python main.py \
  --do_eval_along_training=False \
  --do_predict=True \
  --data_dir=${data_dir} \
  --eval_batch_size=16 \
  --model_dir=${model_dir} \
  --pred_dataset=val