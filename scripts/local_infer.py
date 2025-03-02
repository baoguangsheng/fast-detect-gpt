# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import random

import numpy as np
import torch
import os
import glob
import argparse
import json
from model import load_tokenizer, load_model
from fast_detect_gpt import get_sampling_discrepancy_analytic

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class FastDetectGPT:
    def __init__(self, args):
        self.args = args
        self.criterion_fn = get_sampling_discrepancy_analytic
        self.scoring_tokenizer = load_tokenizer(args.scoring_model_name, args.cache_dir)
        self.scoring_model = load_model(args.scoring_model_name, args.device, args.cache_dir)
        self.scoring_model.eval()
        if args.sampling_model_name != args.scoring_model_name:
            self.sampling_tokenizer = load_tokenizer(args.sampling_model_name, args.cache_dir)
            self.sampling_model = load_model(args.sampling_model_name, args.device, args.cache_dir)
            self.sampling_model.eval()
        # pre-calculated parameters by fitting a LogisticRegression on detection results
        # gpt-j-6B_gpt-neo-2.7B: k: 1.87, b: -2.19, acc: 0.82
        # gpt-neo-2.7B_gpt-neo-2.7B: k: 1.97, b: -1.47, acc: 0.83
        # falcon-7b_falcon-7b-instruct: k: 2.42, b: -2.83, acc: 0.90
        linear_params = {
            'gpt-j-6B_gpt-neo-2.7B': (1.87, -2.19),
            'gpt-neo-2.7B_gpt-neo-2.7B': (1.97, -1.47),
            'falcon-7b_falcon-7b-instruct': (2.42, -2.83),
        }
        key = f'{args.sampling_model_name}_{args.scoring_model_name}'
        self.linear_k, self.linear_b = linear_params[key]

    # compute conditional probability curvature
    def compute_crit(self, text):
        tokenized = self.scoring_tokenizer(text, truncation=True, return_tensors="pt", padding=True, return_token_type_ids=False).to(self.args.device)
        labels = tokenized.input_ids[:, 1:]
        with torch.no_grad():
            logits_score = self.scoring_model(**tokenized).logits[:, :-1]
            if self.args.sampling_model_name == self.args.scoring_model_name:
                logits_ref = logits_score
            else:
                tokenized = self.sampling_tokenizer(text, truncation=True, return_tensors="pt", padding=True, return_token_type_ids=False).to(self.args.device)
                assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
                logits_ref = self.sampling_model(**tokenized).logits[:, :-1]
            crit = self.criterion_fn(logits_ref, logits_score, labels)
        return crit, labels.size(1)

    # compute probability
    def compute_prob(self, text):
        crit, ntoken = self.compute_crit(text)
        prob = sigmoid(self.linear_k * crit + self.linear_b)
        return prob, crit, ntoken


# run interactive local inference
def run(args):
    detector = FastDetectGPT(args)
    # input text
    print('Local demo for Fast-DetectGPT, where the longer text has more reliable result.')
    print('')
    while True:
        print("Please enter your text: (Press Enter twice to start processing)")
        lines = []
        while True:
            line = input()
            if len(line) == 0:
                break
            lines.append(line)
        text = "\n".join(lines)
        if len(text) == 0:
            break
        # estimate the probability of machine generated text
        prob, crit, ntokens = detector.compute_prob(text)
        print(f'Fast-DetectGPT criterion is {crit:.4f}, suggesting that the text has a probability of {prob * 100:.0f}% to be machine-generated.')
        print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # use gpt-neo-2.7B/gpt-neo-2.7B for faster detection
    # use gpt-j-6B/gpt-neo-2.7B for the official setting in the paper
    # use falcon-7b/falcon-7b-instruct for the best detection accuracy
    parser.add_argument('--sampling_model_name', type=str, default="gpt-neo-2.7B")
    parser.add_argument('--scoring_model_name', type=str, default="gpt-neo-2.7B")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--cache_dir', type=str, default="../cache")
    args = parser.parse_args()

    run(args)



