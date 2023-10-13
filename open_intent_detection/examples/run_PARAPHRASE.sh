#!/usr/bin bash

for dataset in 'stackoverflow'
do
    for known_cls_ratio in 1.0
    do
        for labeled_ratio in 1.0
        do
            for seed in 0
            do 
                python gpt_paraphrase.py \
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
                --results_file_name 'results_DA-ADB-BOOST.csv' \
                --save_results \
                --save_model
            done
        done
    done
done

