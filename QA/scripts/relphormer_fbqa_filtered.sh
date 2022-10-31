
for SEED in  111 222 333 444 555 666 777 888 999
do

# echo ${LR} ${WD}
python hitter-bert.py --dataset fbqa \
        --relphormer \
        --filtered \
        --seed ${SEED} \
        --exp_name relphormer-filtered-fbqa \
        --lr 3e-5 \
        --weight_decay 1e-2
done