#!/bin/bash

python2 ../src/segmentor/train_model.py --type train \
--lstm \
--path ../data/ \
--model_save_path ../models/model_test.pkl \
--unigram_embedding ../data/unigram_100_test.embed \
--bigram_embedding  ../data/bigram_100_test.embed \
--max_epoch 100 \
--adam \
--lr 0.001 \

