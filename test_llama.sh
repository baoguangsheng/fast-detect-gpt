#!/usr/bin/env bash

# setup the environment
echo `date`, Setup the environment ...
set -e  # exit if error

# prepare folders
exp_path=exp_gpt3to4
data_path=$exp_path/data
res_path=$exp_path/results
mkdir -p $exp_path $data_path $res_path

datasets="xsum writing pubmed"
source_models="davinci gpt-3.5-turbo gpt-4"

export CUDA_VISIBLE_DEVICES=4

# settings="gpt-j-6B:gpt2-xl gpt-j-6B:gpt-neo-2.7B gpt-j-6B:gpt-j-6B"

# settings="gpt-j-6B:gpt-neo-2.7B"
# settings="vicuna-7b-v1.5:vicuna-7b-v1.5"
settings="llama2-7b:llama2-7b llama2-7b:vicuna-7b-v1.5 vicuna-7b-v1.5:vicuna-7b-v1.5"
for M in $source_models; do
  for D in $datasets; do
    for S in $settings; do
      IFS=':' read -r -a S <<< $S && M1=${S[0]} && M2=${S[1]}
      echo `date`, Evaluating Fast-DetectGPT on ${D}_${M}.${M1}_${M2} ...
      python scripts/fast_detect_gpt.py --reference_model_name $M1 --scoring_model_name $M2 --discrepancy_analytic \
                          --dataset $D --dataset_file $data_path/${D}_${M} --output_file $res_path/${D}_${M}.${M1}_${M2}
    done
  done
done
