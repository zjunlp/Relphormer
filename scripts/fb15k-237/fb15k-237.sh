CHECK_POINT="pretrain/output/epoch=15-Eval/hits10=0.96-Eval/hits1=0.92.ckpt"
PRETRAINED_MODEL_PATH="./Pre-trained_models/bert-base-uncased"
CUDA_VISIBLE_DEVICES=0 nohup python main.py \
   --gpus "1" \
   --max_epochs=5  \
   --num_workers=32 \
   --model_name_or_path  ${PRETRAINED_MODEL_PATH} \
   --accumulate_grad_batches 1 \
   --model_class BertKGC \
   --batch_size 64 \
   --checkpoint ${CHECK_POINT} \
   --pretrain 0 \
   --bce 0 \
   --check_val_every_n_epoch 1 \
   --overwrite_cache \
   --data_dir dataset/fb15k-237 \
   --eval_batch_size 256 \
   --max_seq_length 128 \
   --lr 3e-5 \
   --max_triplet 64 \
   --add_attn_bias True \
   --use_global_node True \
   >logs/train_fb15k-237.log 2>&1 &
