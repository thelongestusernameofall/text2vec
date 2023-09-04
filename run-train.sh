#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

model_name=/data0/research/AI/GPT/Embedding/text2vec-large-chinese-0812-v2
train_file=~/research/AI/GPT/Embedding/train-0817.jsonl
output_dir=~/research/AI/GPT/Embedding/text2vec-large-chinese-0817-v1
num_epochs=3
batch_size=1

python examples/training_sup_text_matching_model_jsonl_data.py \
    --model_arch bert \
    --model_name ${model_name} \
    --train_file ${train_file} \
    --valid_file ~/research/AI/GPT/Embedding/t1.jsonl \
    --test_file ~/research/AI/GPT/Embedding/t1.jsonl \
    --do_train \
    --do_predict \
    --output_dir ${output_dir} \
    --max_seq_length 256 \
    --num_epochs ${num_epochs} \
    --batch_size ${batch_size} \
    --learning_rate 2e-5 
    #--save_model_every_epoch
