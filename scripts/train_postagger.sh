#!/bin/bash

python ./src/postagger/train_model.py \
    --adam \
    --lstm \
    --batch_size 1 \
    --train_path ./outputs/CTB5.1.pos/CTB5.1-train.dat \
    --valid_path ./outputs/CTB5.1.pos/CTB5.1-devel.dat \
    --test_path ./outputs/CTB5.1.pos/CTB5.1-test.dat \
    --word_embedding ./data/embeddings/cn_giga_xin.ctb6_auto_seg.sskip.w5.d100.mc5.ctb51_filtered.embed
