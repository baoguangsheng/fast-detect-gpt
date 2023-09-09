# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import tqdm
import argparse
import json
from data_builder import load_data
from metrics import get_roc_metrics, get_precision_recall_metrics
from model import from_pretrained

def experiment(args):
    # load model
    print(f'Beginning supervised evaluation with {args.model_name}...')
    detector = from_pretrained(AutoModelForSequenceClassification, args.model_name, {}, args.cache_dir).to(args.device)
    tokenizer = from_pretrained(AutoTokenizer, args.model_name, {}, args.cache_dir)
    detector.eval()
    # load data
    data = load_data(args.dataset_file)
    n_samples = len(data["sampled"])
    # eval detector
    name = args.model_name
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    eval_results = []
    for idx in tqdm.tqdm(range(n_samples), desc=f"Computing {name} criterion"):
        original_text = data["original"][idx]
        sampled_text = data["sampled"][idx]
        # original text
        tokenized = tokenizer(original_text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(args.device)
        with torch.no_grad():
            original_crit = detector(**tokenized).logits.softmax(-1)[0, 0].item()
        # sampled text
        tokenized = tokenizer(sampled_text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(args.device)
        with torch.no_grad():
            sampled_crit = detector(**tokenized).logits.softmax(-1)[0, 0].item()
        # result
        eval_results.append({"original": original_text,
                        "original_crit": original_crit,
                        "sampled": sampled_text,
                        "sampled_crit": sampled_crit})

    # compute prediction scores for real/sampled passages
    predictions = {'real': [x["original_crit"] for x in eval_results],
                   'samples': [x["sampled_crit"] for x in eval_results]}
    fpr, tpr, roc_auc = get_roc_metrics(predictions['real'], predictions['samples'])
    p, r, pr_auc = get_precision_recall_metrics(predictions['real'], predictions['samples'])
    print(f"Criterion {name}_threshold ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")
    # log results
    results_file = f'{args.output_file}.{name}.json'
    results = { 'name': f'{name}_threshold',
                'info': {'n_samples': n_samples},
                'predictions': predictions,
                'raw_results': eval_results,
                'metrics': {'roc_auc': roc_auc, 'fpr': fpr, 'tpr': tpr},
                'pr_metrics': {'pr_auc': pr_auc, 'precision': p, 'recall': r},
                'loss': 1 - pr_auc}
    with open(results_file, 'w') as fout:
        json.dump(results, fout)
        print(f'Results written into {results_file}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str, default="./exp_test/results/xsum_gpt2")
    parser.add_argument('--dataset', type=str, default="xsum")
    parser.add_argument('--dataset_file', type=str, default="./exp_test/data/xsum_gpt2")
    parser.add_argument('--model_name', type=str, default="roberta-base-openai-detector")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--cache_dir', type=str, default="../cache")
    args = parser.parse_args()

    experiment(args)
