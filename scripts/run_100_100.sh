#!/bin/bash
data_dir=data/processed_data
LD_PRELOAD=/usr/lib/libtcmalloc.so.4

for experiment_nbr in {1..3}; do
      model_dir="ckpt/consistency/consistency_1.0_1.0_${experiment_nbr}"

      python main.py \
        --do_eval_along_training=True \
        --do_predict=False \
        --sup_cut=1.0 \
        --unsup_cut=1.0 \
        --unsup_ratio=2 \
        --train_batch_size=1 \
        --train_steps=200000 \
        --max_save=1 \
        --data_dir=${data_dir} \
        --model_dir=${model_dir} \
        --unsup_coeff=1 \
        --tsa= \
        --early_stop_steps=30000 \
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
done