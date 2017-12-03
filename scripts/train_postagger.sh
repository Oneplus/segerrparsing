#!/bin/bash

prefix=./outputs/CTB5.1.pos/
python ./src/postagger.py train \
    --optimizer adam \
    --encoder lstm \
    --lr 0.001 \
    --depth 1 \
    --batch_size 32 \
    --model ./models/postagger/ \
    --train_path ${prefix}/CTB5.1-train.dat \
    --valid_path ${prefix}/CTB5.1-devel.dat \
    --test_path ${prefix}/CTB5.1-test.dat \
    --word_embedding ./data/embeddings/cn_giga_xin.ctb6_auto_seg.sskip.w5.d100.mc5.ctb51_filtered.embed
