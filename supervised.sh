#!/usr/bin/env bash
# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# setup the environment
echo `date`, Setup the environment ...
set -e  # exit if error

# prepare folders
exp_path=exp_supervised
data_path=$exp_path/data
res_path=$exp_path/results
mkdir -p $exp_path $data_path $res_path

# preparing dataset
for P in "english:mgpt" "german:mgpt" "pubmed:pubmedgpt" "xsum:gpt2-xl"; do
  IFS=':' read -r -a P <<< $P && D=${P[0]} && M=${P[1]}
  echo `date`, Preparing dataset ${D}-${M} ...
  python scripts/data_builder.py --dataset $D --n_samples 200 --base_model_name $M --output_file $data_path/${D}_${M}
done

# evaluate baselines
for P in "english:mgpt" "german:mgpt" "pubmed:pubmedgpt" "xsum:gpt2-xl"; do
  IFS=':' read -r -a P <<< $P && D=${P[0]} && M=${P[1]}
  echo `date`, Evaluating baseline methods on ${D}_${M} ...
  python scripts/baselines.py --scoring_model_name $M --dataset $D \
                        --dataset_file $data_path/${D}_${M} --output_file $res_path/${D}_${M}
done

# evaluate supervised detectors
for P in "english:mgpt" "german:mgpt" "pubmed:pubmedgpt" "xsum:gpt2-xl"; do
  IFS=':' read -r -a P <<< $P && D=${P[0]} && M=${P[1]}
  for SM in roberta-base-openai-detector roberta-large-openai-detector; do
    echo `date`, Evaluating ${SM} on ${D}_${M} ...
    python scripts/supervised.py --model_name $SM --dataset $D \
                          --dataset_file $data_path/${D}_${M} --output_file $res_path/${D}_${M}
  done
done

# evaluate DetectGPT
for P in "english:mgpt:mt5-xl" "german:mgpt:mt5-xl" "pubmed:pubmedgpt:t5-11b" "xsum:gpt2-xl:t5-11b"; do
  IFS=':' read -r -a P <<< $P && D=${P[0]} && M1=${P[1]} && M2=${P[2]}
  echo `date`, Evaluating DetectGPT on ${D}_${M1}_${M2} ...
  python scripts/detect_gpt.py --scoring_model_name $M1 --mask_filling_model_name $M2 --n_perturbations 100 --dataset $D \
                        --dataset_file $data_path/${D}_${M1} --output_file $res_path/${D}_${M1}_${M2}
done

# evaluate Fast-DetectGPT
for P in "english:mgpt" "german:mgpt" "pubmed:pubmedgpt" "xsum:gpt2-xl"; do
  IFS=':' read -r -a P <<< $P && D=${P[0]} && M=${P[1]}
  echo `date`, Evaluating Fast-DetectGPT on ${D}-${M} ...
  python scripts/fast_detect_gpt.py --sampling_model_name $M --scoring_model_name $M \
                      --dataset $D --dataset_file $data_path/${D}_${M} --output_file $res_path/${D}_${M}
done
