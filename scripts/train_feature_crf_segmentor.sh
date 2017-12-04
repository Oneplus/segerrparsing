#!/bin/bash
for algo in l2sgd lbfgs; do
    for l1 in 0 1; do
        if [[ (${l1} = "1") && (${algo} = "l2sgd") ]]
        then
            continue
        fi
        for l2 in 1e-3 1e-4 1e-5 1e-6; do
            dir=saves/feature_crf_segmentor.${algo}.${l1}.${l2}/
            echo ${dir}
            mkdir -p ${dir}
            python src/feature_crf_segmentor.py \
                --do_train \
                --train ./outputs/CTB5.1.seg/CTB5.1-train.dat \
                --devel ./outputs/CTB5.1.seg/CTB5.1-devel.dat \
                --algorithm ${algo} \
                --l1 ${l1} \
                --l2 ${l2} \
                --output ${dir}/CTB5.1-devel.tag \
                --model ${dir}/model.crfsuite &> ${dir}/train.log

            python src/feature_crf_segmentor.py \
                --devel ./outputs/CTB5.1.seg/CTB5.1-devel.dat \
                --output ${dir}/CTB5.1-devel.tag \
                --model ${dir}/model.crfsuite &> ${dir}/devel.log

            python src/feature_crf_segmentor.py \
                --devel ./outputs/CTB5.1.seg/CTB5.1-test.dat \
                --output ${dir}/CTB5.1-test.tag \
                --model ${dir}/model.crfsuite &> ${dir}/test.log
        done
    done
done
