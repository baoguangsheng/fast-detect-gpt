#!/usr/bin/env bash
# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# setup the environment
echo `date`, Setup the environment ...
set -e  # exit if error

# prepare folders
para=t5  # "t5" for paraphrasing attack, or "random" for decoherence attack
exp_path=exp_attack
data_path=$exp_path/data
res_path=$exp_path/results
mkdir -p $exp_path $data_path $res_path

src_path=exp_gpt3to4
src_data_path=$src_path/data

datasets="xsum writing pubmed"
source_models="gpt-3.5-turbo"

# preparing dataset
for D in $datasets; do
  for M in $source_models; do
    echo `date`, Preparing dataset ${D}_${M} by paraphrasing  ${src_data_path}/${D}_${M} ...
    python scripts/paraphrasing.py --dataset $D --dataset_file $src_data_path/${D}_${M} \
                --paraphraser $para --output_file $data_path/${D}_${M}
  done
done

# evaluate Fast-DetectGPT in the black-box setting
settings="gpt-j-6B:gpt2-xl gpt-j-6B:gpt-neo-2.7B gpt-j-6B:gpt-j-6B"
for D in $datasets; do
  for M in $source_models; do
    for S in $settings; do
      IFS=':' read -r -a S <<< $S && M1=${S[0]} && M2=${S[1]}
      echo `date`, Evaluating Fast-DetectGPT on ${D}_${M}.${M1}_${M2} ...
      python scripts/fast_detect_gpt.py --reference_model_name $M1 --scoring_model_name $M2 --discrepancy_analytic \
                          --dataset $D --dataset_file $data_path/${D}_${M} --output_file $res_path/${D}_${M}.${M1}_${M2}
    done
  done
done

# evaluate supervised detectors
supervised_models="roberta-base-openai-detector roberta-large-openai-detector"
for D in $datasets; do
  for M in $source_models; do
    for SM in $supervised_models; do
      echo `date`, Evaluating ${SM} on ${D}_${M} ...
      python scripts/supervised.py --model_name $SM --dataset $D \
                            --dataset_file $data_path/${D}_${M} --output_file $res_path/${D}_${M}
    done
  done
done

# evaluate fast baselines
scoring_models="gpt-neo-2.7B"
for D in $datasets; do
  for M in $source_models; do
    for M2 in $scoring_models; do
      echo `date`, Evaluating baseline methods on ${D}_${M}.${M2} ...
      python scripts/baselines.py --scoring_model_name ${M2} --dataset $D \
                            --dataset_file $data_path/${D}_${M} --output_file $res_path/${D}_${M}.${M2}
    done
  done
done

# evaluate DetectGPT and DetectLLM
scoring_models="gpt2-xl gpt-neo-2.7B gpt-j-6B"
for D in $datasets; do
  for M in $source_models; do
    M1=t5-11b  # perturbation model
    for M2 in $scoring_models; do
      echo `date`, Evaluating DetectGPT on ${D}_${M}.${M1}_${M2} ...
      python scripts/detect_gpt.py --mask_filling_model_name ${M1} --scoring_model_name ${M2} --n_perturbations 100 --dataset $D \
                          --dataset_file $data_path/${D}_${M} --output_file $res_path/${D}_${M}.${M1}_${M2}
      # we leverage DetectGPT to generate the perturbations
      echo `date`, Evaluating DetectLLM methods on ${D}_${M}.${M1}_${M2} ...
      python scripts/detect_llm.py --scoring_model_name ${M2} --dataset $D \
                          --dataset_file $data_path/${D}_${M}.${M1}.perturbation_100 --output_file $res_path/${D}_${M}.${M1}_${M2}
    done
  done
done
