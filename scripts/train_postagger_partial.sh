#!/bin/bash

python2 ../src/postagger.py train \
    --encoder lstm \
    --train_path ../outputs/partial_data/partial_train.txt \
    --valid_path ../outputs/partial_data/devel.pos \
    --test_path ../outputs/partial_data/test.pos \
    --auto_valid_path ../outputs/partial_data/auto_valid.txt \
    --auto_test_path ../outputs/partial_data/auto_test.txt \
    --gold_valid_path ../outputs/CTB5.1.pos/CTB5.1-devel.dat \
    --gold_test_path ../outputs/CTB5.1.pos/CTB5.1-test.dat \
    --optimizer adam \
    --lr 0.001 \
    --batch_size 16 \asdf
    --model ../models/postagger/partial_model \
    --word_embedding ../data/embeddings/zh.100.embed \
    --max_epoch 10 \
    --use_partial 1 \
