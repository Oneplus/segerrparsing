#!bin/bash

for batch_size in 16 32 64;
do
    for seed in 1 2 3 4 5;
    do
         model_dir=../models/segmentor/segmentor_12_5_${batch_size}_${seed}/
         log_dir=../outputs/log/segmentor_12_5_${batch_size}_${seed}/
         mkdir -p ${model_dir}
         mkdir -p ${log_dir}
         python2 ../src/segmentor.py train \
         --cuda \
         --seed ${seed} \
         --encoder lstm \
         --train_path ../outputs/CTB5.1.seg/CTB5.1-train.dat \
         --valid_path ../outputs/CTB5.1.seg/CTB5.1-devel.dat \
         --test_path ../outputs/CTB5.1.seg/CTB5.1-test.dat \
         --optimizer adam \
         --lr 0.001 \
         --batch_size ${batch_size} \
         --model ${model_dir}/model.pkl \
         --unigram_embedding ../data/embeddings/unigram_100.embed \
         --bigram_embedding  ../data/embeddings/bigram_100.embed \
         --max_epoch 10 &> ${log_dir}/log.txt

    done
done
