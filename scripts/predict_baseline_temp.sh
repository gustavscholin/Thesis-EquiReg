#!/bin/bash
data_dir=data/processed_data
steps=(1577360379 1577075166 1576994269 1577528770 1577443980 1576910133 1577181643 1577264674 1577085243 1577176340 1577259168 1577353181 1576834333 1577621153 1577521850 1577003984 1578654814 1577433742 1577282765 1577395431 1577200599 1577100216 1577480961 1577565887 1577299740 1577220336 1577371843 1577457639 1577123793 1577547231 1576856202 1576927361 1577015129 1576874301 1576952619 1577042311)
model_dirs=(ckpt/baseline/baseline_0.01_3_seed_42 ckpt/baseline/baseline_1.0_1_seed_44 ckpt/baseline/baseline_1.0_1_seed_43 ckpt/baseline/baseline_0.01_3_seed_44 ckpt/baseline/baseline_0.01_3_seed_43 ckpt/baseline/baseline_1.0_1_seed_42 ckpt/baseline/baseline_0.01_2_seed_43 ckpt/baseline/baseline_0.01_2_seed_44 ckpt/baseline/baseline_0.01_2_seed_42 ckpt/baseline/baseline_1.0_2_seed_42 ckpt/baseline/baseline_1.0_2_seed_43 ckpt/baseline/baseline_1.0_2_seed_44 ckpt/baseline/baseline_0.01_1_seed_42 ckpt/baseline/baseline_1.0_3_seed_44 ckpt/baseline/baseline_1.0_3_seed_43 ckpt/baseline/baseline_0.01_1_seed_44 ckpt/baseline/baseline_0.01_1_seed_43 ckpt/baseline/baseline_1.0_3_seed_42 ckpt/baseline/baseline_0.05_2_seed_44 ckpt/baseline/baseline_0.1_3_seed_42 ckpt/baseline/baseline_0.05_2_seed_43 ckpt/baseline/baseline_0.05_2_seed_42 ckpt/baseline/baseline_0.1_3_seed_43 ckpt/baseline/baseline_0.1_3_seed_44 ckpt/baseline/baseline_0.1_2_seed_44 ckpt/baseline/baseline_0.1_2_seed_43 ckpt/baseline/baseline_0.05_3_seed_42 ckpt/baseline/baseline_0.05_3_seed_43 ckpt/baseline/baseline_0.1_2_seed_42 ckpt/baseline/baseline_0.05_3_seed_44 ckpt/baseline/baseline_0.05_1_seed_42 ckpt/baseline/baseline_0.05_1_seed_43 ckpt/baseline/baseline_0.05_1_seed_44 ckpt/baseline/baseline_0.1_1_seed_42 ckpt/baseline/baseline_0.1_1_seed_43 ckpt/baseline/baseline_0.1_1_seed_44)

for i in {30..35}; do

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