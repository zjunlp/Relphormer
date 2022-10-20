nohup python -u main.py --gpus "0," --max_epochs=20  --num_workers=32 \
   --model_name_or_path  bert-base-uncased \
   --accumulate_grad_batches 1 \
   --model_class BertKGC \
   --batch_size 128 \
   --checkpoint /where/you/put/the/checkpoint/file/of/the/pretrain/process \
   --pretrain 0 \
   --bce 0 \
   --check_val_every_n_epoch 1 \
   --overwrite_cache \
   --data_dir dataset/umls \
   --eval_batch_size 256 \
   --max_seq_length 128 \
   --lr 3e-5 \
   --max_triplet 64 \
   --add_attn_bias True \
   --use_global_node True \
   >logs/train_umls.log 2>&1 &

