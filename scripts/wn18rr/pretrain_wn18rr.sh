python main.py --gpus "3," --max_epochs=15  --num_workers=32 \
   --model_name_or_path  "Pre-trained_models/bert-base-uncased" \
   --accumulate_grad_batches 1 \
   --bce 0 \
   --model_class BertKGC \
   --batch_size 128 \
   --pretrain 1 \
   --check_val_every_n_epoch 1 \
   --data_dir dataset/WN18RR \
   --overwrite_cache \
   --eval_batch_size 256 \
   --precision 16 \
   --max_seq_length 32 \
   --lr 3e-5

