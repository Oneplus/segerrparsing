#!bin/bash
for name in 1 2 3 4 5:
do
    model_dir=../models/segmentor/fold_${name}/
    log_dir=../outputs/log/fold/fold_${name}/
    mkdir -p ${model_dir}
    mkdir -p ${log_dir}
    python2 ../src/segmentor.py train \
    --cuda \
    --seed 5 \
    --encoder lstm \
    --train_path ../outputs/fold_${name}/CTB5.1-train.dat \
    --valid_path ../outputs/fold_${name}/CTB5.1-devel.dat \
    --test_path ../outputs/fold_${name}/CTB5.1-test.dat \
    --optimizer adam \
    --lr 0.001 \
    --dropout 0.75 \
    --depth 1 \
    --batch_size 16 \
    --model ${model_dir}/model.pkl \
    --unigram_embedding ../data/embeddings/unigram_100.embed \
    --bigram_embedding  ../data/embeddings/bigram_100.embed \
    --max_epoch 10 &> ${log_dir}/log.txt
done
