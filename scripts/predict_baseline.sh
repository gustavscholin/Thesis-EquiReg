#!/bin/bash
sup_cuts=(0.01 0.05 0.1 1.0)
data_dir=data/processed_data

for experiment_number in {1..3}; do
  for seed in {42..44}; do
    for i in {0..3}; do
      model_dir="/mnt/storage/data/thesis-uda/ckpt/baseline/baseline_${sup_cuts[i]}_${experiment_number}_seed_${seed}"

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