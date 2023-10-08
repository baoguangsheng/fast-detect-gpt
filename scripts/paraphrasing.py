import random

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np
import nltk
from data_builder import load_data, save_data
from model import from_pretrained

class T5Paraphraser:
    def __init__(self, args):
        self.device = args.device
        self.tokenizer = from_pretrained(AutoTokenizer, args.t5_model_name, {}, args.cache_dir)
        self.model = from_pretrained(AutoModelForSeq2SeqLM, args.t5_model_name, {}, args.cache_dir)
        self.model = self.model.to(args.device)
        self.model.eval()

    def paraphrase(self, sents):
        parabatch = ["paraphrase: " + sent + " </s>" for sent in sents]
        encoding = self.tokenizer(parabatch, padding=True, return_tensors="pt")
        input_ids, attention_masks = encoding["input_ids"].to(self.device), encoding["attention_mask"].to(self.device)
        outputs = self.model.generate(
            input_ids=input_ids, attention_mask=attention_masks,
            max_length=256,
            do_sample=True,
            top_k=200,
            top_p=0.95,
            early_stopping=True,
            num_return_sequences=1
        )
        assert len(sents) == len(outputs)
        results = []
        for output, sent in zip(outputs, sents):
            line = self.tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            line = line.strip()
            line = line if len(line) > 0 else sent
            results.append(line)
        return results

class RandomParaphraser:
    def __init__(self, args):
        self.device = args.device

    def paraphrase(self, sents):
        results = []
        for sent in sents:
            words = sent.split()
            if len(words) > 20:
                idx = random.randint(0, len(words) - 2)
                words[idx], words[idx+1] = words[idx+1], words[idx]
            results.append(' '.join(words))
        return results

def generate_data(args):
    data = load_data(args.dataset_file)
    originals = data['original']
    samples = data['sampled']
    print(f"Total number of samples: {len(samples)}")
    print(f"Average number of words: {np.mean([len(x.split()) for x in samples])}")

    if args.do_random_para:
        print(f'Using random paraphraser.')
        paraphraser = RandomParaphraser(args)
    else:
        print(f'Loading model {args.t5_model_name}...')
        paraphraser = T5Paraphraser(args)

    new_samples = []
    for sample in tqdm(samples):
        lines = sample.split('\n')
        new_lines = []
        for line in lines:
            line = line.strip()
            if len(line) == 0:
                new_lines.append(line)
            else:
                sents = nltk.sent_tokenize(line)
                new_sents = paraphraser.paraphrase(sents)
                new_lines.append(' '.join(new_sents))
        new_samples.append('\n'.join(new_lines))

    new_data = {'original': originals, 'sampled': new_samples}
    save_data(args.output_file, args, new_data)


if __name__ == '__main__':
    import argparse
    from tqdm import tqdm
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str, default="./exp_test/results/xsum_gpt2")
    parser.add_argument('--dataset', type=str, default="xsum")
    parser.add_argument('--dataset_file', type=str, default="./exp_test/data/xsum_gpt2")
    parser.add_argument('--t5_model_name', type=str, default="Vamsi/T5_Paraphrase_Paws")
    parser.add_argument('--paraphraser', type=str, default="t5", choices=["t5", "random"])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--cache_dir', type=str, default="../cache")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    import nltk
    nltk.download('punkt')

    generate_data(args)
