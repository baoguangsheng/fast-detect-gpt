# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import matplotlib
import matplotlib.pyplot as plt
import argparse
import glob
import json
from os import path

import numpy as np

matplotlib.use('Agg')

# plot histogram of sampled on left, and original on right
def save_histogram(predictions, figure_file):
    plt.figure(figsize=(4, 2.5))
    plt.subplot(1, 1, 1)
    plt.hist(predictions["samples"], alpha=0.5, bins='auto', label='Model')
    plt.hist(predictions["real"], alpha=0.5, bins='auto', label='Human')
    plt.xlabel("Sampling Discrepancy")
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(figure_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_files', type=str, default="./exp_test/results/*.json")
    parser.add_argument('--draw', action='store_true')
    args = parser.parse_args()

    for res_file in glob.glob(args.result_files, recursive=True):
        with open(res_file, 'r') as fin:
            res = json.load(fin)
        if 'metrics' in res:
            n_samples = res['info']['n_samples']
            roc_auc = res['metrics']['roc_auc']
            real = res['predictions']['real']
            samples = res['predictions']['samples']
            print(f"{res_file}: roc_auc={roc_auc:.4f} n_samples={n_samples} r:{np.mean(real):.2f}/{np.std(real):.2f} s:{np.mean(samples):.2f}/{np.std(samples):.2f}")
        else:
            print(f"{res_file}: metrics not found.")
        # draw histogram
        if args.draw:
            fig_file = f"{res_file}.pdf"
            save_histogram(res['predictions'], fig_file)
            print(f"{fig_file}: histogram figure saved.")

