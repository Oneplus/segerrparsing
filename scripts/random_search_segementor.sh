#!bin/bash

for optimizer in sgd adam;
do
    for lr in 0.01 0.001;
    do
        for batch_size in 1 32;
        do
            for seed in 1 2 3 4 5;
            do
                model_dir=../models/segmentor/segmentor_${optimizer}_${lr}_${batch_size}_${seed}/
                log_dir=../outputs/log/${optimizer}_${lr}_${batch_size}_${seed}/
                mkdir -p ${model_dir}
                mkdir -p ${log_dir}
                python2 ../src/segmentor.py train \
                --cuda \
                --seed ${seed} \
                --encoder lstm \
                --train_path ../outputs/CTB5.1.seg/CTB5.1-train_test.dat \
                --valid_path ../outputs/CTB5.1.seg/CTB5.1-devel_test.dat \
                --test_path ../outputs/CTB5.1.seg/CTB5.1-test_test.dat \
                --optimizer ${optimizer} \
                --lr ${lr} \
                --batch_size ${batch_size} \
                --model ${model_dir}/model.pkl \
                --unigram_embedding ../data/embeddings/unigram_100_test.embed \
                --bigram_embedding  ../data/embeddings/bigram_100_test.embed \
                --max_epoch 1 &> ${log_dir}/log.txt
            done
        done
    done
done







