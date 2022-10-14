python main.py --gpus "1," --max_epochs=5  --num_workers=12 \
   --model_name_or_path  bert-base-uncased \
   --accumulate_grad_batches 1 \
   --model_class BertKGC \
   --batch_size 1 \
   --checkpoint output2/FB15k-237/epoch=9-Eval/hits10=Eval/hits1=0.48-Eval/hits1=0.32.ckpt \
   --pretrain 0 \
   --bce 0 \
   --check_val_every_n_epoch 1 \
   --overwrite_cache \
   --data_dir dataset/FB15k-237 \
   --eval_batch_size 1 \
   --max_seq_length 128 \
   --lr 3e-5 \
   --max_triplet 64 \
   --add_attn_bias True \
   --use_global_node True

