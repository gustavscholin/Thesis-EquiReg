#!/bin/bash
data_dir=data/processed_data
steps=(1581924943 1580961170 1581601449 1582090584 1580882722 1581092658 1581287100 1581166966 1581437114 1581483823 1581339216 1581259944 1581739872 1581193028 1581124503 1581313322 1580992019 1580913290 1580815730 1581391263 1581461798 1581466262 1581387120 1580905279 1580969946 1581061445 1581298083 1581036690 1581118454 1581189872)
model_dirs=(/mnt/storage/data/thesis-uda/ckpt/consistency/consistency_1.0_1.0_2 /mnt/storage/data/thesis-uda/ckpt/consistency/consistency_0.05_1_seed_43 /mnt/storage/data/thesis-uda/ckpt/consistency/consistency_0.05_1_seed_44 /mnt/storage/data/thesis-uda/ckpt/consistency/consistency_1.0_1.0_3 /mnt/storage/data/thesis-uda/ckpt/consistency/consistency_0.05_1_seed_42 /mnt/storage/data/thesis-uda/ckpt/consistency/consistency_0.05_2_seed_42 /mnt/storage/data/thesis-uda/ckpt/consistency/consistency_0.05_2_seed_44 /mnt/storage/data/thesis-uda/ckpt/consistency/consistency_0.05_2_seed_43 /mnt/storage/data/thesis-uda/ckpt/consistency/consistency_0.05_3_seed_43 /mnt/storage/data/thesis-uda/ckpt/consistency/consistency_0.05_3_seed_44 /mnt/storage/data/thesis-uda/ckpt/consistency/consistency_0.05_3_seed_42 /mnt/storage/data/thesis-uda/ckpt/consistency/consistency_0.1_2_seed_44 /mnt/storage/data/thesis-uda/ckpt/consistency/consistency_1.0_1.0_1 /mnt/storage/data/thesis-uda/ckpt/consistency/consistency_0.1_2_seed_43 /mnt/storage/data/thesis-uda/ckpt/consistency/consistency_0.1_2_seed_42 /mnt/storage/data/thesis-uda/ckpt/consistency/consistency_0.1_3_seed_42 /mnt/storage/data/thesis-uda/ckpt/consistency/consistency_0.01_1_seed_44 /mnt/storage/data/thesis-uda/ckpt/consistency/consistency_0.01_1_seed_43 /mnt/storage/data/thesis-uda/ckpt/consistency/consistency_0.01_1_seed_42 /mnt/storage/data/thesis-uda/ckpt/consistency/consistency_0.1_3_seed_43 /mnt/storage/data/thesis-uda/ckpt/consistency/consistency_0.1_3_seed_44 /mnt/storage/data/thesis-uda/ckpt/consistency/consistency_0.01_3_seed_44 /mnt/storage/data/thesis-uda/ckpt/consistency/consistency_0.01_3_seed_43 /mnt/storage/data/thesis-uda/ckpt/consistency/consistency_0.1_1_seed_42 /mnt/storage/data/thesis-uda/ckpt/consistency/consistency_0.1_1_seed_43 /mnt/storage/data/thesis-uda/ckpt/consistency/consistency_0.1_1_seed_44 /mnt/storage/data/thesis-uda/ckpt/consistency/consistency_0.01_3_seed_42 /mnt/storage/data/thesis-uda/ckpt/consistency/consistency_0.01_2_seed_42 /mnt/storage/data/thesis-uda/ckpt/consistency/consistency_0.01_2_seed_43 /mnt/storage/data/thesis-uda/ckpt/consistency/consistency_0.01_2_seed_44)

for i in {0..29}; do

  python main.py \
    --do_eval_along_training=False \
    --do_predict=True \
    --data_dir=${data_dir} \
    --eval_batch_size=16 \
    --model_dir=${model_dirs[i]} \
    --pred_ckpt=${steps[i]} \
    --pred_dataset=val

    python main.py \
    --do_eval_along_training=False \
    --do_predict=True \
    --data_dir=${data_dir} \
    --eval_batch_size=16 \
    --model_dir=${model_dirs[i]} \
    --pred_ckpt=${steps[i]} \
    --pred_dataset=test
done
