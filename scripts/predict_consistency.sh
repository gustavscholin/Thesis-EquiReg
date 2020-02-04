#!/bin/bash
sup_cut=(0.01 0.05 0.1)
unsup_cut=(0.99 0.95 0.9)
data_dir=data/processed_data

for experiment_nbr in {1..3}; do
  for seed in {42..44}; do
    for i in {0..2}; do
      model_dir="/mnt/storage/data/thesis-uda/ckpt/consistency/consistency_${sup_cut[i]}_${unsup_cut[i]}_${experiment_nbr}_seed_${seed}"

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