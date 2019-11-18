python main.py \
  --do_eval_along_training=True \
  --sup_cut=1.0 \
  --unsup_cut=0. \
  --unsup_ratio=0 \
  --shuffle_seed=42 \
  --train_batch_size=2 \
  --train_steps=100000 \
  --data_dir=data/processed_data \
  --model_dir=ckpt/100_supervised