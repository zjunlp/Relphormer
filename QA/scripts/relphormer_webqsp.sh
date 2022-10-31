
for SEED in  222 333 444 555 666 777 888 999
do
python hitter-bert.py --dataset webqsp \
        --relphormer \
        --seed ${SEED} \
        --exp_name relphormer-webqsp \
        --lr 3e-5 \
        --weight_decay 1e-2
done