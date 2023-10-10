# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os.path
import argparse
import json
import numpy as np


def save_lines(lines, file):
    with open(file, 'w') as fout:
        fout.write('\n'.join(lines))

def get_auroc(result_file):
    with open(result_file, 'r') as fin:
        res = json.load(fin)
        return res['metrics']['roc_auc']

def get_fpr_tpr(result_file):
    with open(result_file, 'r') as fin:
        res = json.load(fin)
        return res['metrics']['fpr'], res['metrics']['tpr']

def report_main_results(args):
    datasets = {'xsum': 'XSum',
                'squad': 'SQuAD',
                'writing': 'WritingPrompts'}
    source_models = {'gpt2-xl': 'GPT-2',
                     'opt-2.7b': 'OPT-2.7',
                     'gpt-neo-2.7B': 'Neo-2.7',
                     'gpt-j-6B': 'GPT-J',
                     'gpt-neox-20b': 'NeoX'}
    methods1 = {'likelihood': 'Likelihood',
               'entropy': 'Entropy',
               'logrank': 'LogRank',
               'lrr': 'LRR',
               'npr': 'NPR'}
    methods2 = {'perturbation_100': 'DetectGPT',
               'sampling_discrepancy': 'Fast-DetectGPT'}

    def _get_method_aurocs(dataset, method, filter=''):
        cols = []
        for model in source_models:
            result_file = f'{args.result_path}/{dataset}_{model}{filter}.{method}.json'
            if os.path.exists(result_file):
                auroc = get_auroc(result_file)
            else:
                auroc = 0.0
            cols.append(auroc)
        cols.append(np.mean(cols))
        return cols

    headers = ['Method'] + [source_models[model] for model in source_models] + ['Avg.']
    for dataset in datasets:
        print('----')
        print(datasets[dataset])
        print('----')
        print(' '.join(headers))
        # basic methods
        for method in methods1:
            method_name = methods1[method]
            cols = _get_method_aurocs(dataset, method)
            cols = [f'{col:.4f}' for col in cols]
            print(method_name, ' '.join(cols))
        # white-box comparison
        results = {}
        for method in methods2:
            method_name = methods2[method]
            cols = _get_method_aurocs(dataset, method)
            results[method_name] = cols
            cols = [f'{col:.4f}' for col in cols]
            print(method_name, ' '.join(cols))
        cols = np.array(results['Fast-DetectGPT']) - np.array(results['DetectGPT'])
        cols = [f'{col:.4f}' for col in cols]
        print('(Diff)', ' '.join(cols))
        # black-box comparison
        filters = {'perturbation_100': '.t5-3b_gpt-neo-2.7B',
                    'sampling_discrepancy': '.gpt-j-6B_gpt-neo-2.7B'}
        results = {}
        for method in methods2:
            method_name = methods2[method]
            cols = _get_method_aurocs(dataset, method, filters[method])
            results[method_name] = cols
            cols = [f'{col:.4f}' for col in cols]
            print(method_name, ' '.join(cols))
        cols = np.array(results['Fast-DetectGPT']) - np.array(results['DetectGPT'])
        cols = [f'{col:.4f}' for col in cols]
        print('(Diff)', ' '.join(cols))

def report_main_ext_results(args):
    datasets = {'xsum': 'XSum',
                'squad': 'SQuAD',
                'writing': 'WritingPrompts'}
    source_models = {'bloom-7b1': 'BLOOM-7.1',
                     'opt-13b': 'OPT-13',
                     'llama-13b': 'Llama-13',
                     'llama2-13b': 'Llama2-13',
                     }
    methods1 = {'likelihood': 'Likelihood',
               'entropy': 'Entropy',
               'logrank': 'LogRank',
               'lrr': 'LRR',
               'npr': 'NPR'}
    methods2 = {'perturbation_100': 'DetectGPT',
               'sampling_discrepancy': 'Fast-DetectGPT'}

    def _get_method_aurocs(dataset, method, filter=''):
        cols = []
        for model in source_models:
            result_file = f'{args.result_path}/{dataset}_{model}{filter}.{method}.json'
            if os.path.exists(result_file):
                auroc = get_auroc(result_file)
            else:
                auroc = 0.0
            cols.append(auroc)
        cols.append(np.mean(cols))
        return cols

    headers = ['Method'] + [source_models[model] for model in source_models] + ['Avg.']
    for dataset in datasets:
        print('----')
        print(datasets[dataset])
        print('----')
        print(' '.join(headers))
        # basic methods
        for method in methods1:
            method_name = methods1[method]
            cols = _get_method_aurocs(dataset, method)
            cols = [f'{col:.4f}' for col in cols]
            print(method_name, ' '.join(cols))
        # white-box comparison
        results = {}
        for method in methods2:
            method_name = methods2[method]
            cols = _get_method_aurocs(dataset, method)
            results[method_name] = cols
            cols = [f'{col:.4f}' for col in cols]
            print(method_name, ' '.join(cols))
        cols = np.array(results['Fast-DetectGPT']) - np.array(results['DetectGPT'])
        cols = [f'{col:.4f}' for col in cols]
        print('(Diff)', ' '.join(cols))
        # black-box comparison
        filters = {'perturbation_100': '.t5-3b_gpt-neo-2.7B',
                    'sampling_discrepancy': '.gpt-j-6B_gpt-neo-2.7B'}
        results = {}
        for method in methods2:
            method_name = methods2[method]
            cols = _get_method_aurocs(dataset, method, filters[method])
            results[method_name] = cols
            cols = [f'{col:.4f}' for col in cols]
            print(method_name, ' '.join(cols))
        cols = np.array(results['Fast-DetectGPT']) - np.array(results['DetectGPT'])
        cols = [f'{col:.4f}' for col in cols]
        print('(Diff)', ' '.join(cols))

def report_refmodel_results(args):
    datasets = {'xsum': 'XSum',
                'squad': 'SQuAD',
                'writing': 'WritingPrompts'}
    source_models = {'gpt2-xl': 'GPT-2',
                     'gpt-neo-2.7B': 'Neo-2.7',
                     'gpt-j-6B': 'GPT-J'}

    def _get_method_aurocs(method, ref_model=None):
        cols = []
        for dataset in datasets:
            for model in source_models:
                filter = '' if ref_model is None or ref_model == model else f'.{ref_model}_{model}'
                result_file = f'{args.result_path}/{dataset}_{model}{filter}.{method}.json'
                if os.path.exists(result_file):
                    auroc = get_auroc(result_file)
                else:
                    auroc = 0.0
                cols.append(auroc)
        cols.append(np.mean(cols))
        return cols

    headers1 = ['----'] + list([datasets[d] for d in datasets])
    headers2 = ['Method'] + [source_models[model] for model in source_models] \
              + [source_models[model] for model in source_models] \
              + [source_models[model] for model in source_models] \
              + ['Avg.']
    print(' '.join(headers1))
    print(' '.join(headers2))

    ref_models = [None, 'gpt2-xl', 'gpt-neo-2.7B', 'gpt-j-6B']
    for ref_model in ref_models:
        method = 'sampling_discrepancy'
        method_name = 'Fast-DetectGPT (*/*)' if ref_model is None else f'Fast-DetectGPT ({source_models[ref_model]}/*)'
        cols = _get_method_aurocs(method, ref_model)
        cols = [f'{col:.4f}' for col in cols]
        print(method_name, ' '.join(cols))


def report_chatgpt_gpt4_results(args):
    datasets = {'xsum': 'XSum',
                'writing': 'Writing',
                'pubmed': 'PubMed'}
    source_models = {'gpt-3.5-turbo': 'ChatGPT',
                     'gpt-4': 'GPT-4'}
    score_models = { 't5-11b': 'T5-11B',
                     'gpt2-xl': 'GPT-2',
                     'opt-2.7b': 'OPT-2.7',
                     'gpt-neo-2.7B': 'Neo-2.7',
                     'gpt-j-6B': 'GPT-J',
                     'gpt-neox-20b': 'NeoX'}
    methods1 = {'roberta-base-openai-detector': 'RoBERTa-base',
                'roberta-large-openai-detector': 'RoBERTa-large'}
    methods2 = {'likelihood': 'Likelihood', 'entropy': 'Entropy', 'logrank': 'LogRank'}
    methods3 = {'lrr': 'LRR', 'npr': 'NPR', 'perturbation_100': 'DetectGPT',
                'sampling_discrepancy_analytic': 'Fast'}

    def _get_method_aurocs(method, filter=''):
        results = []
        for model in source_models:
            cols = []
            for dataset in datasets:
                result_file = f'{args.result_path}/{dataset}_{model}{filter}.{method}.json'
                if os.path.exists(result_file):
                    auroc = get_auroc(result_file)
                else:
                    auroc = 0.0
                cols.append(auroc)
            cols.append(np.mean(cols))
            results.extend(cols)
        return results

    headers1 = ['--'] + [source_models[model] for model in source_models]
    headers2 = ['Method'] + [datasets[dataset] for dataset in datasets] + ['Avg.'] \
               + [datasets[dataset] for dataset in datasets] + ['Avg.']
    print(' '.join(headers1))
    print(' '.join(headers2))
    # supervised methods
    for method in methods1:
        method_name = methods1[method]
        cols = _get_method_aurocs(method)
        cols = [f'{col:.4f}' for col in cols]
        print(method_name, ' '.join(cols))
    # zero-shot methods

    filters2 = {'likelihood': ['.gpt2-xl', '.gpt-neo-2.7B', '.gpt-j-6B', '.gpt-neox-20b'],
               'entropy': ['.gpt2-xl', '.gpt-neo-2.7B', '.gpt-j-6B', '.gpt-neox-20b'],
               'logrank': ['.gpt2-xl', '.gpt-neo-2.7B', '.gpt-j-6B', '.gpt-neox-20b']}
    filters3 = {'lrr': ['.t5-11b_gpt2-xl', '.t5-11b_gpt-neo-2.7B', '.t5-11b_gpt-j-6B', '.t5-11b_gpt-neox-20b'],
               'npr': ['.t5-11b_gpt2-xl', '.t5-11b_gpt-neo-2.7B', '.t5-11b_gpt-j-6B', '.t5-11b_gpt-neox-20b'],
               'perturbation_100': ['.t5-11b_gpt2-xl', '.t5-11b_gpt-neo-2.7B', '.t5-11b_gpt-j-6B', '.t5-11b_gpt-neox-20b'],
               'sampling_discrepancy_analytic': ['.gpt-j-6B_gpt2-xl', '.gpt-j-6B_gpt-neo-2.7B', '.gpt-j-6B_gpt-j-6B', '.gpt-neox-20b_gpt-neox-20b']}
    for method in methods2:
        for filter in filters2[method]:
            setting = score_models[filter[1:]]
            method_name = f'{methods2[method]}({setting})'
            cols = _get_method_aurocs(method, filter)
            cols = [f'{col:.4f}' for col in cols]
            print(method_name, ' '.join(cols))
    for method in methods3:
        for filter in filters3[method]:
            setting = [score_models[model] for model in filter[1:].split('_')]
            method_name = f'{methods3[method]}({setting[0]}/{setting[1]})'
            cols = _get_method_aurocs(method, filter)
            cols = [f'{col:.4f}' for col in cols]
            print(method_name, ' '.join(cols))

def report_gpt3_results(args):
    datasets = {'xsum': 'XSum',
                'writing': 'Writing',
                'pubmed': 'PubMed'}
    source_models = {'davinci': 'GPT-3'}
    score_models = { 't5-11b': 'T5-11B',
                     'gpt2-xl': 'GPT-2',
                     'opt-2.7b': 'OPT-2.7',
                     'gpt-neo-2.7B': 'Neo-2.7',
                     'gpt-j-6B': 'GPT-J',
                     'gpt-neox-20b': 'NeoX'}
    methods1 = {'roberta-base-openai-detector': 'RoBERTa-base',
                'roberta-large-openai-detector': 'RoBERTa-large'}
    methods2 = {'likelihood': 'Likelihood', 'entropy': 'Entropy', 'logrank': 'LogRank'}
    methods3 = {'lrr': 'LRR', 'npr': 'NPR', 'perturbation_100': 'DetectGPT',
                'sampling_discrepancy_analytic': 'Fast'}

    def _get_method_aurocs(method, filter=''):
        results = []
        for model in source_models:
            cols = []
            for dataset in datasets:
                result_file = f'{args.result_path}/{dataset}_{model}{filter}.{method}.json'
                if os.path.exists(result_file):
                    auroc = get_auroc(result_file)
                else:
                    auroc = 0.0
                cols.append(auroc)
            cols.append(np.mean(cols))
            results.extend(cols)
        return results

    headers1 = ['--'] + [source_models[model] for model in source_models]
    headers2 = ['Method'] + [datasets[dataset] for dataset in datasets] + ['Avg.'] \
               + [datasets[dataset] for dataset in datasets] + ['Avg.']
    print(' '.join(headers1))
    print(' '.join(headers2))
    # supervised methods
    for method in methods1:
        method_name = methods1[method]
        cols = _get_method_aurocs(method)
        cols = [f'{col:.4f}' for col in cols]
        print(method_name, ' '.join(cols))
    # zero-shot methods

    filters2 = {'likelihood': ['.gpt2-xl', '.gpt-neo-2.7B', '.gpt-j-6B', '.gpt-neox-20b'],
               'entropy': ['.gpt2-xl', '.gpt-neo-2.7B', '.gpt-j-6B', '.gpt-neox-20b'],
               'logrank': ['.gpt2-xl', '.gpt-neo-2.7B', '.gpt-j-6B', '.gpt-neox-20b']}
    filters3 = {'lrr': ['.t5-11b_gpt2-xl', '.t5-11b_gpt-neo-2.7B', '.t5-11b_gpt-j-6B', '.t5-11b_gpt-neox-20b'],
               'npr': ['.t5-11b_gpt2-xl', '.t5-11b_gpt-neo-2.7B', '.t5-11b_gpt-j-6B', '.t5-11b_gpt-neox-20b'],
               'perturbation_100': ['.t5-11b_gpt2-xl', '.t5-11b_gpt-neo-2.7B', '.t5-11b_gpt-j-6B', '.t5-11b_gpt-neox-20b'],
               'sampling_discrepancy_analytic': ['.gpt-j-6B_gpt2-xl', '.gpt-j-6B_gpt-neo-2.7B', '.gpt-j-6B_gpt-j-6B', '.gpt-neox-20b_gpt-neox-20b']}
    for method in methods2:
        for filter in filters2[method]:
            setting = score_models[filter[1:]]
            method_name = f'{methods2[method]}({setting})'
            cols = _get_method_aurocs(method, filter)
            cols = [f'{col:.4f}' for col in cols]
            print(method_name, ' '.join(cols))
    for method in methods3:
        for filter in filters3[method]:
            setting = [score_models[model] for model in filter[1:].split('_')]
            method_name = f'{methods3[method]}({setting[0]}/{setting[1]})'
            cols = _get_method_aurocs(method, filter)
            cols = [f'{col:.4f}' for col in cols]
            print(method_name, ' '.join(cols))

def report_maxlen_trends(args):
    datasets = {'xsum': 'XSum',
                'writing': 'WritingPrompts'}
    source_models = {'gpt-3.5-turbo': 'ChatGPT',
                     'gpt-4': 'GPT-4'}
    score_models = {'t5-11b': 'T5-11B',
                    'gpt2-xl': 'GPT-2',
                    'opt-2.7b': 'OPT-2.7',
                    'gpt-neo-2.7B': 'Neo-2.7',
                    'gpt-j-6B': 'GPT-J',
                    'gpt-neox-20b': 'NeoX'}
    methods1 = {'roberta-base-openai-detector': 'RoBERTa-base',
                'roberta-large-openai-detector': 'RoBERTa-large'}
    methods2 = {'likelihood': 'Likelihood'}
    methods3 = {'perturbation_100': 'DetectGPT',
                'sampling_discrepancy_analytic': 'Fast-Detect'}
    maxlens = [30, 60, 90, 120, 150, 180]

    def _get_method_aurocs(root_path, dataset, source_model, method, filter=''):
        cols = []
        for maxlen in maxlens:
            result_file = f'{root_path}/exp_maxlen{maxlen}/results/{dataset}_{source_model}{filter}.{method}.json'
            if os.path.exists(result_file):
                auroc = get_auroc(result_file)
            else:
                auroc = 0.0
            cols.append(auroc)
        return cols

    filters2 = {'likelihood': '.gpt-neo-2.7B'}
    filters3 = {'perturbation_100': '.t5-11b_gpt-neo-2.7B',
                'sampling_discrepancy_analytic': '.gpt-j-6B_gpt-neo-2.7B'}

    headers = ['Method'] + [str(maxlen) for maxlen in maxlens]
    print(' '.join(headers))
    # print table per model and dataset
    results = {}
    for model in source_models:
        model_name = source_models[model]
        for data in datasets:
            data_name = datasets[data]
            print('----')
            print(f'{model_name} / {data_name}')
            print('----')
            for method in methods1:
                method_name = methods1[method]
                cols = _get_method_aurocs('.', data, model, method)
                results[f'{model_name}_{data_name}_{method_name}'] = cols
                cols = [f'{col:.4f}' for col in cols]
                print(method_name, ' '.join(cols))
            for method in methods2:
                filter = filters2[method]
                setting = score_models[filter[1:]]
                method_name = f'{methods2[method]}({setting})'
                cols = _get_method_aurocs('.', data, model, method, filter)
                results[f'{model_name}_{data_name}_{method_name}'] = cols
                cols = [f'{col:.4f}' for col in cols]
                print(method_name, ' '.join(cols))
            for method in methods3:
                filter = filters3[method]
                setting = [score_models[model] for model in filter[1:].split('_')]
                method_name = f'{methods3[method]}({setting[0]}/{setting[1]})'
                cols = _get_method_aurocs('.', data, model, method, filter)
                results[f'{model_name}_{data_name}_{method_name}'] = cols
                cols = [f'{col:.4f}' for col in cols]
                print(method_name, ' '.join(cols))
    import json
    json_file = './exp_analysis/maxlen_trends.json'
    with open(json_file, 'w') as fout:
        json.dump(results, fout)
        print(f'Write to file {json_file}')

def report_auroc_curve(args):
    datasets = {'xsum': 'XSum',
                'writing': 'WritingPrompts'}
    source_models = {'gpt-3.5-turbo': 'ChatGPT',
                     'gpt-4': 'GPT-4'}
    score_models = {'t5-11b': 'T5-11B',
                    'gpt2-xl': 'GPT-2',
                    'opt-2.7b': 'OPT-2.7',
                    'gpt-neo-2.7B': 'Neo-2.7',
                    'gpt-j-6B': 'GPT-J',
                    'gpt-neox-20b': 'NeoX'}
    methods1 = {'roberta-base-openai-detector': 'RoBERTa-base',
                'roberta-large-openai-detector': 'RoBERTa-large'}
    methods2 = {'likelihood': 'Likelihood'}
    methods3 = {'perturbation_100': 'DetectGPT',
                'sampling_discrepancy_analytic': 'Fast-Detect'}

    def _get_method_fpr_tpr(root_path, dataset, source_model, method, filter=''):
        maxlen = 180
        result_file = f'{root_path}/exp_maxlen{maxlen}/results/{dataset}_{source_model}{filter}.{method}.json'
        if os.path.exists(result_file):
            fpr, tpr = get_fpr_tpr(result_file)
        else:
            fpr, tpr = [], []
        assert len(fpr) == len(tpr)
        return list(zip(fpr, tpr))

    filters2 = {'likelihood': '.gpt-neo-2.7B'}
    filters3 = {'perturbation_100': '.t5-11b_gpt-neo-2.7B',
                'sampling_discrepancy_analytic': '.gpt-j-6B_gpt-neo-2.7B'}

    # print table per model and dataset
    results = {}
    for model in source_models:
        model_name = source_models[model]
        for data in datasets:
            data_name = datasets[data]
            print('----')
            print(f'{model_name} / {data_name}')
            print('----')
            for method in methods1:
                method_name = methods1[method]
                cols = _get_method_fpr_tpr('.', data, model, method)
                results[f'{model_name}_{data_name}_{method_name}'] = cols
                cols = [f'({col[0]:.3f},{col[1]:.3f})' for col in cols]
                print(method_name, ' '.join(cols))
            for method in methods2:
                filter = filters2[method]
                setting = score_models[filter[1:]]
                method_name = f'{methods2[method]}({setting})'
                cols = _get_method_fpr_tpr('.', data, model, method, filter)
                results[f'{model_name}_{data_name}_{method_name}'] = cols
                cols = [f'({col[0]:.3f},{col[1]:.3f})' for col in cols]
                print(method_name, ' '.join(cols))
            for method in methods3:
                filter = filters3[method]
                setting = [score_models[model] for model in filter[1:].split('_')]
                method_name = f'{methods3[method]}({setting[0]}/{setting[1]})'
                cols = _get_method_fpr_tpr('.', data, model, method, filter)
                results[f'{model_name}_{data_name}_{method_name}'] = cols
                cols = [f'({col[0]:.3f},{col[1]:.3f})' for col in cols]
                print(method_name, ' '.join(cols))
    import json
    json_file = './exp_analysis/auroc_curve.json'
    with open(json_file, 'w') as fout:
        json.dump(results, fout)
        print(f'Write to file {json_file}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path', type=str, default="./exp_main/results/")
    parser.add_argument('--report_name', type=str, default="main_results")
    args = parser.parse_args()

    if args.report_name == 'main_results':
        report_main_results(args)
    elif args.report_name == 'main_ext_results':
        report_main_ext_results(args)
    elif args.report_name == 'chatgpt_gpt4_results':
        report_chatgpt_gpt4_results(args)
    elif args.report_name == 'gpt3_results':
        report_gpt3_results(args)
    elif args.report_name == 'maxlen_trends':
        report_maxlen_trends(args)
    elif args.report_name == 'auroc_curve':
        report_auroc_curve(args)
    elif args.report_name == 'refmodel_results':
        report_refmodel_results(args)