#!/bin/bash
python main.py \
  --do_eval_along_training=False \
  --do_predict=True \
  --data_dir=data/processed_data \
  --eval_batch_size=16 \
  $@
