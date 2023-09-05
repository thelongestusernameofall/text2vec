#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

model_name=/data0/research/AI/GPT/Embedding/text2vec-large-chinese-0817-v1
train_file=~/research/AI/GPT/Embedding/all-0904-clean.jsonl
#train_file=~/research/AI/GPT/Embedding/train-0904.jsonl
valid_file=~/research/AI/GPT/Embedding/train-0904.jsonl
output_dir=~/research/AI/GPT/Embedding/text2vec-large-chinese-0905
num_epochs=4
batch_size=200
lr=2e-4

python examples/training_sup_text_matching_model_jsonl_data.py \
    --model_arch sentencebert \
    --model_name ${model_name} \
    --train_file ${train_file} \
    --valid_file ${valid_file} \
    --test_file ${valid_file} \
    --do_train \
    --do_predict \
    --output_dir ${output_dir} \
    --max_seq_length 256 \
    --num_epochs ${num_epochs} \
    --batch_size ${batch_size} \
    --data_parallel \
    --learning_rate ${lr}
    #--save_model_every_epoch
    #--data_parallel \
