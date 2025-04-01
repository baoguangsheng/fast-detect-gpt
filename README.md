# Fast-DetectGPT
**This code is for ICLR 2024 paper "Fast-DetectGPT: Efficient Zero-Shot Detection of Machine-Generated Text via Conditional Probability Curvature"**, where we borrow or extend some code from [DetectGPT](https://github.com/eric-mitchell/detect-gpt).

[Paper](https://arxiv.org/abs/2310.05130) 
| [LocalDemo](#local-demo)
| [OnlineDemo](https://aidetect.lab.westlake.edu.cn/)
| [OpenReview](https://openreview.net/forum?id=Bpcgcr8E8Z)

* :fire: API support is launched. Please check the [API page](https://aidetect.lab.westlake.edu.cn/#/apidoc) in the demo.
* :fire: Fast-DetectGPT can utilize GPT-3.5 and other proprietary models as its scoring model now via [Glimpse](https://github.com/baoguangsheng/glimpse).
* :fire: So far the best sampling/scoring models we found for Fast-DetectGPT are falcon-7b/falcon-7b-instruct.

## Brief Intro
<table class="tg"  style="padding-left: 30px;">
  <tr>
    <th class="tg-0pky">Method</th>
    <th class="tg-0pky">5-Model Generations ↑</th>
    <th class="tg-0pky">ChatGPT/GPT-4 Generations ↑</th>
    <th class="tg-0pky">Speedup ↑</th>
  </tr>
  <tr>
    <td class="tg-0pky">DetectGPT</td>
    <td class="tg-0pky">0.9554</td>
    <td class="tg-0pky">0.7225</td>
    <td class="tg-0pky">1x</td>
  </tr>
  <tr>
    <td class="tg-0pky">Fast-DetectGPT</td>
    <td class="tg-0pky">0.9887 (relative↑ <b>74.7%</b>)</td>
    <td class="tg-0pky">0.9338 (relative↑ <b>76.1%</b>)</td>
    <td class="tg-0pky"><b>340x</b></td>
  </tr>
</table>
The table shows detection accuracy (measured in AUROC) and computational speedup for machine-generated text detection. The <b>white-box setting</b> (directly using the source model) is used for detecting generations produced by five source models (5-model), whereas the <b>black-box
setting</b> (utilizing surrogate models) targets ChatGPT and GPT-4 generations. AUROC results are averaged across various datasets and source models. Speedup assessments were conducted on a Tesla A100 GPU.


## Environment
* Python3.8
* PyTorch1.10.0
* Setup the environment:
  ```bash setup.sh```
  
(Notes: our experiments are run on 1 GPU of Tesla A100 with 80G memory.)

## Local Demo
Please run following command locally for an interactive demo:
```
python scripts/local_infer.py
```
where the default sampling and scoring models are both gpt-neo-2.7B.

We could use gpt-j-6B as the sampling model to obtain more accurate detections:
```
python scripts/local_infer.py  --sampling_model_name gpt-j-6B
```


An example (using gpt-j-6B as the sampling model) looks like
```
Please enter your text: (Press Enter twice to start processing)
Disguised as police, they broke through a fence on Monday evening and broke into the cargo of a Swiss-bound plane to take the valuable items. The audacious heist occurred at an airport in a small European country, leaving authorities baffled and airline officials in shock.

Fast-DetectGPT criterion is 1.9299, suggesting that the text has a probability of 82% to be machine-generated.
```

## Workspace
Following folders are created for our experiments:
* ./exp_main -> experiments for 5-model generations (main.sh).
* ./exp_gpt3to4 -> experiments for GPT-3, ChatGPT, and GPT-4 generations (gpt3to4.sh).

(Notes: we share <b>generations from GPT-3, ChatGPT, and GPT-4</b> in exp_gpt3to4/data for convenient reproduction.)

### Citation
If you find this work useful, you can cite it with the following BibTex entry:

    @inproceedings{bao2023fast,
      title={Fast-DetectGPT: Efficient Zero-Shot Detection of Machine-Generated Text via Conditional Probability Curvature},
      author={Bao, Guangsheng and Zhao, Yanbin and Teng, Zhiyang and Yang, Linyi and Zhang, Yue},
      booktitle={The Twelfth International Conference on Learning Representations},
      year={2023}
    }

