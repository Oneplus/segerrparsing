#!/bin/bash

python2 ../src/postagger.py train \
    --encoder lstm \
    --train_path ../data/pos/pos_partial_train.txt \
    --valid_path ../data/pos/pos_partial_devel.txt \
    --test_path ../data/pos/pos_partial_test.txt \
    --optimizer adam \
    --lr 0.001 \
    --batch_size 32 \
    --model ../models/postagger/model_1.pkl \
    --word_embedding ../data/embeddings/zh.100-test.embed \
    --max_epoch 1 \
    --use_partial 0 \
