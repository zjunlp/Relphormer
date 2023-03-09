DATA_DIR="/home/bz/Relphormer-github/dataset/fb15k-237"
nohup python -u main.py --gpus "1" --max_epochs=16  --num_workers=32 \
   --model_name_or_path  bert-base-uncased \
   --accumulate_grad_batches 1 \
   --model_class BertKGC \
   --batch_size 256 \
   --pretrain 1 \
   --bce 0 \
   --check_val_every_n_epoch 1 \
   --overwrite_cache \
   --data_dir ${DATA_DIR} \
   --eval_batch_size 256 \
   --max_seq_length 64 \
   --lr 1e-4 \
   >logs/pretrain_fb15k-237.log 2>&1 &