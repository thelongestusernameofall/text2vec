#!/bin/bash
python examples/training_sup_text_matching_model_jsonl_data.py \
    --model_arch bert \
    --model_name ~/research/AI/GPT/Embedding/text2vec-large-chinese \
    --train_file ~/research/AI/GPT/Embedding/t1.jsonl \
    --valid_file ~/research/AI/GPT/Embedding/t1.jsonl \
    --test_file ~/research/AI/GPT/Embedding/t1.jsonl \
    --do_train \
    --output_dir ~/research/AI/GPT/Embedding/text2vec-large-chinese-t1 \
    --max_seq_length 256 \
    --num_epochs 10 \
    --batch_size 1 \
    --learning_rate 2e-5