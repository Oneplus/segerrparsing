#!/bin/bash

mkdir -p ./outputs/CTB5.1.pos/
for fold in train devel test;
do
    python ./src/postagger/conll_to_dat.py \
        --input ./data/CTB5.1/dependency-penn2malt/CTB5.1-${fold}.gp.conll \
        --output ./outputs/CTB5.1.pos/CTB5.1-${fold}.dat
done
