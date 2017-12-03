#!/usr/bin/env bash
dir=saves/feature_crf_postagger/
mkdir -p ${dir}
python src/feature_crf_postagger.py --train ./outputs/CTB5.1.pos/CTB5.1-train.dat \
--devel ./outputs/CTB5.1.pos/CTB5.1-test.dat \
--output ${dir}/CTB5.1-test.tag \
--model ${dir}/model.crfsuite