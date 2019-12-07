#!/bin/bash

HOME=/home/chojunghee/code/GAN
PROG=$HOME/gan.py

list_batch_size='32'
dataset='cauchy'
datapath='../../data/noise2'
epoch='200'
li='1e-4'
lf='1e-4' 
mt='0.5'
wd='0.0'
iter='1'
moreInfo='gan_'$dataset

for bs in $list_batch_size; do
    CUDA_VISIBLE_DEVICES=$1 python -u $PROG --dataset $dataset --data_path $datapath --epochs $epoch --batch_size $bs \
    --lr_initial $li --lr_final $lf --iteration $iter --momentum $mt --weight_decay $wd --moreInfo $moreInfo
done