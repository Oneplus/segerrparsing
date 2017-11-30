#!/bin/bash

python2 ../src/segmentor/train_model.py --type test \
--lstm \
--path ../data/ \
--model_save_path ../models/model.pkl \
--unigram_embedding ../data/unigram_100_test.embed \
--bigram_embedding  ../data/bigram_100_test.embed \
--res_path ../output/test_res.txt \
