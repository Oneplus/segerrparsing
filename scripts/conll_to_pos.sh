#!/bin/bash

mkdir -p ./outputs/CTB5.1.pos/
for fold in train devel test;
do
    python ./src/conll_to_posdat.py \
        --input ./data/CTB5.1/dependency-penn2malt/CTB5.1-${fold}.gp.conll \
        --output ./outputs/CTB5.1.pos/CTB5.1-${fold}.dat
done

echo "done."
