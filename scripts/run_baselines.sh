#!/bin/bash
sup_cuts=(0.01 0.05 0.1 1.0)
data_dir=data/processed_data
save_steps=(125 250 250 500)
# dec_steps=(2000 5000 7000 50000)
# min_steps=(0 0 12500 50000)
early_stop_steps=(2000 10000 10000 20000)

for experiment_number in {1..3}; do
  for seed in {42..44}; do
    for i in {0..3}; do
      model_dir="ckpt/baseline/baseline_${sup_cuts[i]}_${experiment_number}_seed_${seed}"

      python main.py \
        --do_eval_along_training=True \
        --do_predict=False \
        --sup_cut=${sup_cuts[i]} \
        --unsup_cut=0. \
        --unsup_ratio=0 \
        --shuffle_seed=${seed} \
        --train_batch_size=4 \
        --train_steps=100000 \
        --save_steps=${save_steps[i]} \
        --max_save=1 \
        --data_dir=${data_dir} \
        --model_dir=${model_dir} \
        --early_stop_steps=${early_stop_steps[i]}  \
        --min_step=0 \
        --exp_lr_decay=False

      python main.py \
        --do_eval_along_training=False \
        --do_predict=True \
        --data_dir=${data_dir} \
        --eval_batch_size=16 \
        --model_dir=${model_dir} \
        --pred_dataset=val
    done
  done
done