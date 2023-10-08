# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import time

import numpy as np
import datasets
import torch
import random
import argparse
import os
import json
import custom_datasets
from model import load_tokenizer, load_model

def stats_str(data):
    if type(data) == dict:
        mean_orig = np.mean([len(v.split()) for v in data['original']])
        mean_samp = np.mean([len(v.split()) for v in data['sampled']])
        return f'{mean_orig:.0f} words (original), {mean_samp:.0f} words (sampled).'
    else:
        mean_orig = np.mean([len(v['original'].split()) for v in data])
        mean_samp = np.mean([len(v['sampled'].split()) for v in data])
        mean_perturb_orig = np.mean([np.mean([len(p.split()) for p in v['perturbed_original']]) for v in data])
        mean_perturb_samp = np.mean([np.mean([len(p.split()) for p in v['perturbed_sampled']]) for v in data])
        return f'{mean_orig:.0f} words (original), {mean_samp:.0f} words (sampled), {mean_perturb_orig:.0f} words (perturb original), {mean_perturb_samp:.0f} words (perturb sampled).'

def save_data(output_file, args, data):
    # write args to file
    args_file = f"{output_file}.args.json"
    with open(args_file, "w") as fout:
        json.dump(args, fout, indent=4)
        print(f"Args written into {args_file}")

    # write the data to a json file in the save folder
    data_file = f"{output_file}.raw_data.json"
    with open(data_file, "w") as fout:
        json.dump(data, fout, indent=4)
        print(f"Raw data written into {data_file}: {stats_str(data)}")


def load_data(input_file):
    # load args from file
    args_file = f"{input_file}.args.json"
    with open(args_file, "r") as fin:
        args = json.load(fin)
        print(f"Args loaded from {args_file}")

    # load the data from file
    data_file = f"{input_file}.raw_data.json"
    with open(data_file, "r") as fin:
        data = json.load(fin)
        print(f"Raw data loaded from {data_file}: {stats_str(data)}")

    return args, data

def convert_data(input_file, output_file, max_words):
    def _reduce(text):
        lines = []
        nwords = 0
        for line in text.split('\n'):
            if nwords >= max_words:
                break
            words = line.split()
            words = words[:max_words - nwords]
            lines.append(' '.join(words))
            nwords += len(words)
        return '\n'.join(lines)

    args, data = load_data(input_file)
    if type(data) == dict:
        data['original'] = [_reduce(x) for x in data['original']]
        data['sampled'] = [_reduce(x) for x in data['sampled']]
    else:
        for item in data:
            item['original'] = _reduce(item['original'])
            item['sampled'] = _reduce(item['sampled'])
            item['perturbed_original'] = [_reduce(x) for x in item['perturbed_original']]
            item['perturbed_sampled'] = [_reduce(x) for x in item['perturbed_sampled']]

    save_data(output_file, args, data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default="./exp_gpt3to4/data/")
    parser.add_argument('--output_path', type=str, default="./exp_maxlen150/data/")
    parser.add_argument('--max_words', type=int, default=150)
    args = parser.parse_args()

    import glob
    import os.path as path

    for file_name in glob.glob(f'{args.input_path}/*.raw_data.json'):
        print(file_name)
        file_name = path.basename(file_name).replace('.raw_data.json', '')
        convert_data(path.join(args.input_path, file_name), path.join(args.output_path, file_name), args.max_words)
