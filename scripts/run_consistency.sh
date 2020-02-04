#!/bin/bash
sup_cut=(0.01 0.05 0.1)
# unsup_cut=(0.99 0.95 0.9)
# min_steps=(0 50000 50000)
data_dir=data/processed_data
early_stop_steps=(10000 20000 30000)

for experiment_nbr in {1..3}; do
  for seed in {42..44}; do
    for i in {0..2}; do
      model_dir="/mnt/storage/data/thesis-uda/ckpt/consistency/consistency_${sup_cut[i]}_${experiment_nbr}_seed_${seed}"

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
        --early_stop_steps=${early_stop_steps[i]} \
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
    done
  done
done
