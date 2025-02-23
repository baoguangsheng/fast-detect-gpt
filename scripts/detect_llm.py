# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import argparse
import json
from model import load_tokenizer, load_model
from metrics import get_roc_metrics, get_precision_recall_metrics
from data_builder import load_data

def get_likelihood(logits, labels):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1

    logits = logits.view(-1, logits.shape[-1])
    labels = labels.view(-1)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    log_likelihood = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return log_likelihood.mean().item()

def get_logrank(logits, labels):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1

    # get rank of each label token in the model's likelihood ordering
    matches = (logits.argsort(-1, descending=True) == labels.unsqueeze(-1)).nonzero()
    assert matches.shape[1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"

    ranks, timesteps = matches[:, -1], matches[:, -2]

    # make sure we got exactly one match for each timestep in the sequence
    assert (timesteps == torch.arange(len(timesteps)).to(timesteps.device)).all(), "Expected one match per timestep"

    ranks = ranks.float() + 1  # convert to 1-indexed rank
    ranks = torch.log(ranks)
    return ranks.mean().item()

# Log-Likelihood Log-Rank Ratio
def get_lrr(args, scoring_model, scoring_tokenizer, text, perturbs):
    with torch.no_grad():
        tokenized = scoring_tokenizer(text, return_tensors="pt", return_token_type_ids=False).to(args.device)
        labels = tokenized.input_ids[:, 1:]
        logits = scoring_model(**tokenized).logits[:, :-1]
        likelihood = get_likelihood(logits, labels)
        logrank = get_logrank(logits, labels)
        return - likelihood / logrank

# Normalized Log-Rank Perturbation
def get_npr(args, scoring_model, scoring_tokenizer, text, perturbs):
    with torch.no_grad():
        tokenized = scoring_tokenizer(text, return_tensors="pt", return_token_type_ids=False).to(args.device)
        labels = tokenized.input_ids[:, 1:]
        logits = scoring_model(**tokenized).logits[:, :-1]
        logrank = get_logrank(logits, labels)
        # perturbations
        logranks = []
        for perturb in perturbs:
            tokenized = scoring_tokenizer(perturb, return_tensors="pt", return_token_type_ids=False).to(args.device)
            labels = tokenized.input_ids[:, 1:]
            logits = scoring_model(**tokenized).logits[:, :-1]
            logranks.append(get_logrank(logits, labels))
        # npr
        return np.mean(logranks) / logrank

def experiment(args):
    # load model
    scoring_tokenizer = load_tokenizer(args.scoring_model_name, args.cache_dir)
    scoring_model = load_model(args.scoring_model_name, args.device, args.cache_dir)
    scoring_model.eval()
    # load data
    data = load_data(args.dataset_file)
    n_samples = len(data)
    # eval criterions
    criterion_fns = {'lrr': get_lrr, 'npr': get_npr}
    for name in criterion_fns:
        criterion_fn = criterion_fns[name]
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        eval_results = []
        for idx in tqdm.tqdm(range(n_samples), desc=f"Computing {name} criterion"):
            original_text = data[idx]["original"]
            sampled_text = data[idx]["sampled"]
            perturbed_original = data[idx]["perturbed_original"]
            perturbed_sampled = data[idx]["perturbed_sampled"]
            original_crit = criterion_fn(args, scoring_model, scoring_tokenizer, original_text, perturbed_original)
            sampled_crit = criterion_fn(args, scoring_model, scoring_tokenizer, sampled_text, perturbed_sampled)
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
    parser.add_argument('--dataset_file', type=str, default="./exp_test/results/xsum_gpt2.perturbation_10")
    parser.add_argument('--scoring_model_name', type=str, default="gpt2")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--cache_dir', type=str, default="../cache")
    args = parser.parse_args()

    experiment(args)
