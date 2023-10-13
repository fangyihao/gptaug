#!/usr/bin bash

for dataset in 'stackoverflow_cg'
do
    for known_cls_ratio in 0.25 0.5 0.75
    do
        for labeled_ratio in 1.0
        do
            for seed in 9
            do 
                python run.py \
                --dataset $dataset \
                --method 'DA-ADB' \
                --known_cls_ratio $known_cls_ratio \
                --labeled_ratio $labeled_ratio \
                --seed $seed \
                --backbone 'bert_disaware_boost' \
                --config_file_name 'DA-ADB' \
                --loss_fct 'CrossEntropyLoss' \
                --gpu_id '0' \
                --train \
                --pretrain \
                --results_file_name 'results_DA-ADB-BOOST-F-10.csv' \
                --save_results
            done
        done
    done
done

