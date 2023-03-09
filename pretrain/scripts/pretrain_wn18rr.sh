DATA_DIR="/home/bz/Relphormer-github/dataset/wn18rr"
nohup python -u main.py --gpus "1" --max_epochs=15  --num_workers=32 \
   --model_name_or_path  bert-base-uncased \
   --accumulate_grad_batches 1 \
   --bce 0 \
   --model_class BertKGC \
   --batch_size 512 \
   --pretrain 1 \
   --check_val_every_n_epoch 1 \
   --data_dir ${DATA_DIR} \
   --overwrite_cache \
   --eval_batch_size 256 \
   --max_seq_length 32 \
   --lr 1e-4 \
   >logs/pretrain_wn18rr.log 2>&1 &
   # --precision 16 \
