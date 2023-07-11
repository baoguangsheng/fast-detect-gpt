# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import glob
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_files', type=str, default="./exp_test/results/*.json")
    args = parser.parse_args()

    for res_file in glob.glob(args.result_files, recursive=True):
        with open(res_file, 'r') as fin:
            res = json.load(fin)
        if 'metrics' in res:
            n_samples = res['info']['n_samples']
            roc_auc = res['metrics']['roc_auc']
            print(f"{res_file}: roc_auc={roc_auc:.4f} n_samples={n_samples}")
        else:
            print(f"{res_file}: metrics not found.")
