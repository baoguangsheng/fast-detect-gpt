# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import time

import numpy as np
import tqdm
import argparse
import json
from metrics import get_roc_metrics, get_precision_recall_metrics
from data_builder import load_data

def detect_gptzero(args, text):
    import requests
    url = "https://api.gptzero.me/v2/predict/text"
    payload = {
        "document": text,
        "version": "2023-09-14"
    }
    headers = {
        "Accept": "application/json",
        "content-type": "application/json",
        "x-api-key": ""
    }

    while True:
        try:
            time.sleep(600)  # 1 request per 10 minutes for free access
            response = requests.post(url, json=payload, headers=headers)
            return response.json()['documents'][0]['completely_generated_prob']
        except Exception as ex:
            print(ex)

def experiment(args):
    # load data
    data = load_data(args.dataset_file)
    n_samples = len(data["sampled"])
    # evaluate criterion
    name = "gptzero"
    criterion_fn = detect_gptzero

    results = []
    for idx in tqdm.tqdm(range(n_samples), desc=f"Computing {name} criterion"):
        original_text = data["original"][idx]
        sampled_text = data["sampled"][idx]
        original_crit = criterion_fn(args, original_text)
        sampled_crit = criterion_fn(args, sampled_text)
        # result
        results.append({"original": original_text,
                        "original_crit": original_crit,
                        "sampled": sampled_text,
                        "sampled_crit": sampled_crit})

        # compute prediction scores for real/sampled passages
        predictions = {'real': [x["original_crit"] for x in results],
                       'samples': [x["sampled_crit"] for x in results]}
        print(f"Real mean/std: {np.mean(predictions['real']):.2f}/{np.std(predictions['real']):.2f}, Samples mean/std: {np.mean(predictions['samples']):.2f}/{np.std(predictions['samples']):.2f}")
        fpr, tpr, roc_auc = get_roc_metrics(predictions['real'], predictions['samples'])
        p, r, pr_auc = get_precision_recall_metrics(predictions['real'], predictions['samples'])
        print(f"Criterion {name}_threshold ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")

    # results
    results_file = f'{args.output_file}.{name}.json'
    results = { 'name': f'{name}_threshold',
                'info': {'n_samples': n_samples},
                'predictions': predictions,
                'raw_results': results,
                'metrics': {'roc_auc': roc_auc, 'fpr': fpr, 'tpr': tpr},
                'pr_metrics': {'pr_auc': pr_auc, 'precision': p, 'recall': r},
                'loss': 1 - pr_auc}
    with open(results_file, 'w') as fout:
        json.dump(results, fout)
        print(f'Results written into {results_file}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str, default="./exp_gpt3to4/results/xsum_gpt-4")
    parser.add_argument('--dataset', type=str, default="xsum")
    parser.add_argument('--dataset_file', type=str, default="./exp_gpt3to4/data/xsum_gpt-4")
    args = parser.parse_args()

    experiment(args)

