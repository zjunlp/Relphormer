python main.py --gpus "3," --max_epochs=20  --num_workers=32 \
   --model_name_or_path  bert-base-uncased \
   --accumulate_grad_batches 1 \
   --model_class BertKGC \
   --batch_size 128 \
   --checkpoint output/WN18RR/epoch=14-step=49699-Eval/hits10=0.90.ckpt \
   --pretrain 0 \
   --bce 0 \
   --check_val_every_n_epoch 5 \
   --overwrite_cache \
   --data_dir dataset/WN18RR \
   --eval_batch_size 256 \
   --max_seq_length 128 \
   --lr 3e-5 \
   --max_triplet 32 \
   --add_attn_bias True \
   --use_global_node True

