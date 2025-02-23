# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os.path

import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import re
import torch
import tqdm
import argparse
import json
from data_builder import load_data, save_data
from metrics import get_roc_metrics, get_precision_recall_metrics
from model import load_tokenizer, load_model, get_model_fullname, from_pretrained

# define regex to match all <extra_id_*> tokens, where * is an integer
pattern = re.compile(r"<extra_id_\d+>")

def load_mask_model(model_name, device, cache_dir):
    model_name = get_model_fullname(model_name)
    # mask filling t5 model
    print(f'Loading mask filling model {model_name}...')
    mask_model = from_pretrained(AutoModelForSeq2SeqLM, model_name, {}, cache_dir)
    mask_model = mask_model.to(device)
    return mask_model

def load_mask_tokenizer(model_name, max_length, cache_dir):
    model_name = get_model_fullname(model_name)
    tokenizer = from_pretrained(AutoTokenizer, model_name, {'model_max_length': max_length}, cache_dir)
    return tokenizer

def tokenize_and_mask(text, span_length, pct, ceil_pct=False):
    buffer_size = 1
    tokens = text.split(' ')
    mask_string = '<<<mask>>>'

    n_spans = pct * len(tokens) / (span_length + buffer_size * 2)
    if ceil_pct:
        n_spans = np.ceil(n_spans)
    n_spans = int(n_spans)

    n_masks = 0
    while n_masks < n_spans:
        start = np.random.randint(0, len(tokens) - span_length)
        end = start + span_length
        search_start = max(0, start - buffer_size)
        search_end = min(len(tokens), end + buffer_size)
        if mask_string not in tokens[search_start:search_end]:
            tokens[start:end] = [mask_string]
            n_masks += 1

    # replace each occurrence of mask_string with <extra_id_NUM>, where NUM increments
    num_filled = 0
    for idx, token in enumerate(tokens):
        if token == mask_string:
            tokens[idx] = f'<extra_id_{num_filled}>'
            num_filled += 1
    assert num_filled == n_masks, f"num_filled {num_filled} != n_masks {n_masks}"
    text = ' '.join(tokens)
    return text

def count_masks(texts):
    return [len([x for x in text.split() if x.startswith("<extra_id_")]) for text in texts]

# replace each masked span with a sample from T5 mask_model
def replace_masks(args, mask_model, mask_tokenizer, texts):
    n_expected = count_masks(texts)
    stop_id = mask_tokenizer.encode(f"<extra_id_{max(n_expected)}>")[0]
    tokens = mask_tokenizer(texts, return_tensors="pt", padding=True).to(args.device)
    outputs = mask_model.generate(**tokens, max_length=150, do_sample=True, top_p=args.mask_top_p,
                                  num_return_sequences=1, eos_token_id=stop_id)
    return mask_tokenizer.batch_decode(outputs, skip_special_tokens=False)

def extract_fills(texts):
    # remove <pad> from beginning of each text
    texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in texts]

    # return the text in between each matched mask token
    extracted_fills = [pattern.split(x)[1:-1] for x in texts]

    # remove whitespace around each fill
    extracted_fills = [[y.strip() for y in x] for x in extracted_fills]

    return extracted_fills

def apply_extracted_fills(masked_texts, extracted_fills):
    # split masked text into tokens, only splitting on spaces (not newlines)
    tokens = [x.split(' ') for x in masked_texts]

    n_expected = count_masks(masked_texts)

    # replace each mask token with the corresponding fill
    for idx, (text, fills, n) in enumerate(zip(tokens, extracted_fills, n_expected)):
        if len(fills) < n:
            tokens[idx] = []
        else:
            for fill_idx in range(n):
                text[text.index(f"<extra_id_{fill_idx}>")] = fills[fill_idx]

    # join tokens back into text
    texts = [" ".join(x) for x in tokens]
    return texts

def perturb_texts_(args, mask_model, mask_tokenizer, texts, ceil_pct=False):
    span_length = args.span_length
    pct = args.pct_words_masked
    masked_texts = [tokenize_and_mask(x, span_length, pct, ceil_pct) for x in texts]
    raw_fills = replace_masks(args, mask_model, mask_tokenizer, masked_texts)
    extracted_fills = extract_fills(raw_fills)
    perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)

    # Handle the fact that sometimes the model doesn't generate the right number of fills and we have to try again
    attempts = 1
    while '' in perturbed_texts:
        idxs = [idx for idx, x in enumerate(perturbed_texts) if x == '']
        print(f'WARNING: {len(idxs)} texts have no fills. Trying again [attempt {attempts}].')
        masked_texts = [tokenize_and_mask(x, span_length, pct, ceil_pct) for idx, x in enumerate(texts) if idx in idxs]
        raw_fills = replace_masks(args, mask_model, mask_tokenizer, masked_texts)
        extracted_fills = extract_fills(raw_fills)
        new_perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)
        for idx, x in zip(idxs, new_perturbed_texts):
            perturbed_texts[idx] = x
        attempts += 1
    return perturbed_texts

def perturb_texts(args, mask_model, mask_tokenizer, texts, ceil_pct=False):
    chunk_size = 10
    outputs = []
    for i in range(0, len(texts), chunk_size):
        outputs.extend(perturb_texts_(args, mask_model, mask_tokenizer, texts[i:i + chunk_size], ceil_pct=ceil_pct))
    return outputs

# Get the log likelihood of each text under the base_model
def get_ll(args, scoring_model, scoring_tokenizer, text):
    with torch.no_grad():
        tokenized = scoring_tokenizer(text, return_tensors="pt", return_token_type_ids=False).to(args.device)
        labels = tokenized.input_ids
        return -scoring_model(**tokenized, labels=labels).loss.item()

def get_lls(args, scoring_model, scoring_tokenizer, texts):
    return [get_ll(args, scoring_model, scoring_tokenizer, text) for text in texts]


def generate_perturbs(args):
    n_perturbations = args.n_perturbations
    name = f'perturbation_{n_perturbations}'
    # load model
    mask_model = load_mask_model(args.mask_filling_model_name, args.device, args.cache_dir)
    mask_model.eval()
    try:
        n_positions = mask_model.config.n_positions
    except AttributeError:
        n_positions = 512
    mask_tokenizer = load_mask_tokenizer(args.mask_filling_model_name, n_positions, args.cache_dir)

    # load data
    data = load_data(args.dataset_file)
    n_samples = len(data["sampled"])

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # generate perturb samples
    perturbs = []
    for idx in tqdm.tqdm(range(n_samples), desc=f"Perturb text"):
        original_text = data["original"][idx]
        sampled_text = data["sampled"][idx]
        # perturb
        p_sampled_text = perturb_texts(args, mask_model, mask_tokenizer, [sampled_text for _ in range(n_perturbations)])
        p_original_text = perturb_texts(args, mask_model, mask_tokenizer, [original_text for _ in range(n_perturbations)])
        assert len(p_sampled_text) == n_perturbations, f"Expected {n_perturbations} perturbed samples, got {len(p_sampled_text)}"
        assert len(p_original_text) == n_perturbations, f"Expected {n_perturbations} perturbed samples, got {len(p_original_text)}"
        # result
        perturbs.append({
            "original": original_text,
            "sampled": sampled_text,
            "perturbed_sampled": p_sampled_text,
            "perturbed_original": p_original_text
        })

    save_data(f'{args.dataset_file}.{args.mask_filling_model_name}.{name}', args, perturbs)


def experiment(args):
    n_perturbations = args.n_perturbations
    name = f'perturbation_{n_perturbations}'
    perturb_file = f'{args.dataset_file}.{args.mask_filling_model_name}.{name}.raw_data.json'
    if os.path.exists(perturb_file):
        print(f'Use existing perturbation file: {perturb_file}')
    else:
        generate_perturbs(args)
    # load model
    scoring_tokenizer = load_tokenizer(args.scoring_model_name, args.cache_dir)
    scoring_model = load_model(args.scoring_model_name, 'cpu', args.cache_dir)
    scoring_model.eval()
    scoring_model.to(args.device)
    # load data
    data = load_data(f'{args.dataset_file}.{args.mask_filling_model_name}.{name}')
    n_samples = len(data)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Evaluate
    results = data
    for idx in tqdm.tqdm(range(n_samples), desc=f"Computing {name} criterion"):
        original_text = results[idx]["original"]
        sampled_text = results[idx]["sampled"]
        perturbed_original = results[idx]["perturbed_original"]
        perturbed_sampled = results[idx]["perturbed_sampled"]
        # original text
        original_ll = get_ll(args, scoring_model, scoring_tokenizer, original_text)
        p_original_ll = get_lls(args, scoring_model, scoring_tokenizer, perturbed_original)
        # sampled text
        sampled_ll = get_ll(args, scoring_model, scoring_tokenizer, sampled_text)
        p_sampled_ll = get_lls(args, scoring_model, scoring_tokenizer, perturbed_sampled)
        # result
        results[idx]["original_ll"] = original_ll
        results[idx]["sampled_ll"] = sampled_ll
        results[idx]["all_perturbed_sampled_ll"] = p_sampled_ll
        results[idx]["all_perturbed_original_ll"] = p_original_ll
        results[idx]["perturbed_sampled_ll"] = np.mean(p_sampled_ll)
        results[idx]["perturbed_original_ll"] = np.mean(p_original_ll)
        results[idx]["perturbed_sampled_ll_std"] = np.std(p_sampled_ll) if len(p_sampled_ll) > 1 else 1
        results[idx]["perturbed_original_ll_std"] = np.std(p_original_ll) if len(p_original_ll) > 1 else 1

    # compute diffs with perturbed
    predictions = {'real': [], 'samples': []}
    for res in results:
        if res['perturbed_original_ll_std'] == 0:
            res['perturbed_original_ll_std'] = 1
            print("WARNING: std of perturbed original is 0, setting to 1")
            print(f"Number of unique perturbed original texts: {len(set(res['perturbed_original']))}")
            print(f"Original text: {res['original']}")
        if res['perturbed_sampled_ll_std'] == 0:
            res['perturbed_sampled_ll_std'] = 1
            print("WARNING: std of perturbed sampled is 0, setting to 1")
            print(f"Number of unique perturbed sampled texts: {len(set(res['perturbed_sampled']))}")
            print(f"Sampled text: {res['sampled']}")
        predictions['real'].append((res['original_ll'] - res['perturbed_original_ll']) / res['perturbed_original_ll_std'])
        predictions['samples'].append((res['sampled_ll'] - res['perturbed_sampled_ll']) / res['perturbed_sampled_ll_std'])

    print(f"Real mean/std: {np.mean(predictions['real']):.2f}/{np.std(predictions['real']):.2f}, Samples mean/std: {np.mean(predictions['samples']):.2f}/{np.std(predictions['samples']):.2f}")
    fpr, tpr, roc_auc = get_roc_metrics(predictions['real'], predictions['samples'])
    p, r, pr_auc = get_precision_recall_metrics(predictions['real'], predictions['samples'])
    print(f"Criterion {name}_threshold ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")

    # results
    results_file = f'{args.output_file}.{name}.json'
    results = {
        'name': name,
        'info': {
            'pct_words_masked': args.pct_words_masked,
            'span_length': args.span_length,
            'n_perturbations': args.n_perturbations,
            'n_samples': n_samples,
        },
        'predictions': predictions,
        'raw_results': results,
        'metrics': {
            'roc_auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr,
        },
        'pr_metrics': {
            'pr_auc': pr_auc,
            'precision': p,
            'recall': r,
        },
        'loss': 1 - pr_auc,
    }
    with open(results_file, 'w') as fout:
        json.dump(results, fout)
        print(f'Results written into {results_file}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str, default="./exp_test/results/xsum_gpt2")
    parser.add_argument('--dataset', type=str, default="xsum")
    parser.add_argument('--dataset_file', type=str, default="./exp_test/data/xsum_gpt2")
    parser.add_argument('--pct_words_masked', type=float, default=0.3) # pct masked is actually pct_words_masked * (span_length / (span_length + 2 * buffer_size))
    parser.add_argument('--mask_top_p', type=float, default=1.0)
    parser.add_argument('--span_length', type=int, default=2)
    parser.add_argument('--n_perturbations', type=int, default=10)
    parser.add_argument('--scoring_model_name', type=str, default="gpt2")
    parser.add_argument('--mask_filling_model_name', type=str, default="t5-small")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--cache_dir', type=str, default="../cache")
    args = parser.parse_args()

    experiment(args)
