#!/bin/bash

for number in 1 2 3 4 5
do
    python ../src/ulits/bies_to_sentence.py \
    --input '../outputs/res_seg/fold_${number}/CTB5.1-test.segtag' \
    --output '../outputs/res_seg/fold_${number}/CTB5.1-test.sentence' \
done
