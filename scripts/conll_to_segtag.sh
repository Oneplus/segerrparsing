#!/bin/bash
prefix=outputs
mkdir -p ../${prefix}/CTB5.1.seg/
for fold in train devel test;
do
    python ../src/conll_to_segdat.py \
        --input ../data/CTB5.1/CTB5.1-${fold}.gp.conll \
        --output ../${prefix}/CTB5.1.seg/CTB5.1-${fold}.dat
done
echo "done."
