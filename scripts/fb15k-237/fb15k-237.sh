CHECK_POINT="/home/bz/Relphormer-github/pretrain/output/FB15k-237/epoch=15-step=19299-Eval/hits10=0.96.ckpt"
nohup python -u main.py \
   --gpus "1" \
   --max_epochs=16  \
   --num_workers=32 \
   --model_name_or_path  bert-base-uncased \
   --accumulate_grad_batches 1 \
   --model_class BertKGC \
   --batch_size 200 \
   --checkpoint ${CHECK_POINT} \
   --pretrain 0 \
   --bce 0 \
   --check_val_every_n_epoch 1 \
   --overwrite_cache \
   --data_dir dataset/fb15k-237 \
   --eval_batch_size 200 \
   --max_seq_length 128 \
   --lr 3e-5 \
   --max_triplet 64 \
   --add_attn_bias True \
   --use_global_node True \
   >logs/train_fb15k-237.log 2>&1 &
