import re
import os
from datasets import Dataset, load_dataset, concatenate_datasets
from random import randint, seed, choice
from typing import List, Tuple
from tqdm import tqdm
from verl.utils.hdfs_io import copy, makedirs
import argparse
import math

def chr2num(char: str) -> int:
    return int(char) if char.isdigit() else ord(char) - ord('a') + 10

def make_prefix(dp, base_range, template_type):
    S, T, B = dp['S'], dp['T'], dp['B']
    b_min, b_max = base_range
    if template_type == 'base':
        """This works for any base model"""
        prefix = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
User: Given the number string "{S}" and its decimal value {T}, determine the base it is written in. The base must be an integer from {b_min} to {b_max} (inclusive). Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> 2 </answer>.
Assistant: Let me solve this step by step.
<think>"""
    elif template_type == 'qwen-instruct':
        """This works for Qwen Instruct Models"""
        prefix = f"""<|im_start|>system\nYou are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.<|im_end|>\n<|im_start|>user\nGiven the number string "{S}" and its decimal value {T}, determine the base it is written in. The base must be an integer from {b_min} to {b_max} (inclusive). Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> 2 </answer>.<|im_end|>\n<|im_start|>assistant\nLet me solve this step by step.\n<think>"""
    return prefix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hf_url', default='sdpkjc/NumBase-N01-S28-B28')
    parser.add_argument('--local_dir', default='./numbase')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--num_samples', type=int, default=100000)
    parser.add_argument('--train_size', type=int, default=327680)
    parser.add_argument('--test_size', type=int, default=1024)
    parser.add_argument('--template_type', type=str, default='base')

    args = parser.parse_args()

    data_source = 'numbase'
    TRAIN_SIZE = args.train_size
    TEST_SIZE = args.test_size

    raw_dataset = load_dataset(args.hf_url, split='train')

    n_repeat = math.ceil(1 / min(1, (len(raw_dataset) / (TRAIN_SIZE + TEST_SIZE))))
    repeated_dataset = concatenate_datasets([raw_dataset] * n_repeat).shuffle()

    assert len(repeated_dataset) > TRAIN_SIZE + TEST_SIZE
    train_dataset = repeated_dataset.select(range(TRAIN_SIZE))
    test_dataset = repeated_dataset.select(range(TRAIN_SIZE, TRAIN_SIZE + TEST_SIZE))

    base_range = (chr2num(args.hf_url[-2]), chr2num(args.hf_url[-1]))    
    def make_map_fn(split):
        def process_fn(example, idx):
            question = make_prefix(example, base_range=base_range, template_type=args.template_type)
            solution = {
                "S": example['S'],
                "T": example['T'],
                "B": example['B'],
            }
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            return data
        return process_fn
    
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = os.path.join(args.local_dir, args.hf_url.split('/')[-1])
    print(local_dir)

    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir) 
