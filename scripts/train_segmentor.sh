#!/bin/bash

python2 ../src/segmentor.py train \
    --encoder lstm \
    --train_path ../outputs/CTB5.1.seg/CTB5.1-train.dat \
    --valid_path ../outputs/CTB5.1.seg/CTB5.1-devel.dat \
    --test_path ../outputs/CTB5.1.seg/CTB5.1-test.dat \
    --optimizer adam \
    --lr 0.001 \
    --batch_size 32 \
    --model ../models/segmentor/model_1.pkl \
    --unigram_embedding ../data/embeddings/unigram_100_test.embed \
    --bigram_embedding  ../data/embeddings/bigram_100_test.embed \
    --max_epoch 1
