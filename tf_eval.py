import os, re, sys, copy, gzip, glob
import argparse
import torch
import transformers
import numpy as np
import pandas as pd
import random

import evaluate
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from tqdm.auto import tqdm
from datasets import Dataset
from process import process_fasta

from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType
from collections import Counter
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S',
                    stream=sys.stdout)

def pad_batch(batch, device):
    b = {}
    max_length = max([len(b) for b in batch['input_ids']])
    for f in batch.features:
        if f=='labels':
            b[f] = batch[f][:].to(device)
        else:
            b[f] = torch.vstack([torch.cat((b, torch.zeros(max_length - len(b), dtype=b.dtype))) for b in batch[f]]).to(device)
    return b

def eval_model(model, eval_data, batch_size, metric, progress_bar=None):
    model.eval()
    out_probs, etalons = [], []
    for batch_id in range(int(np.ceil(len(eval_data) / batch_size))):
        to_index = min(len(eval_data), (batch_id+1)*batch_size)
        batch = pad_batch(eval_data.select(range(batch_id*batch_size, to_index)), model.device)
        with torch.no_grad():
            outputs = model(**batch)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        out_probs.extend(probs[:,1].tolist())
        etalons.extend(batch['labels'].tolist())
        predictions = torch.argmax(probs, dim=-1)
        metric.add_batch(predictions=predictions, references=batch['labels'])
        if progress_bar is not None:
            progress_bar.update(1)
    return metric.compute(), out_probs, etalons

def process_data(data, tokenizer, samples_to_use=-1, shuffle=True):
    def tokenize(examples):
        return tokenizer(examples['sequence'], padding=False, truncation=True)

    data = data.shuffle(seed=42) if shuffle else data

    if samples_to_use < 0 or samples_to_use > len(data):
        samples_to_use = len(data)
    elif samples_to_use < 1:
        samples_to_use = int(samples_to_use * len(data))
    data = data.select(range(samples_to_use))
    data = data.map(tokenize, batched=True)

    to_keep = set(('labels input_ids attention_mask token_type_ids').split())
    data = data.remove_columns([col for col in data.features if col not in to_keep])
    data.set_format("torch")
    return data

def main():
    parser = argparse.ArgumentParser(description='Finetuning experiments.')

    parser.add_argument('--tf_id', default=155580)
    parser.add_argument('--eval_file', default='03.TF_binding_table_Phchr4_2.csv')
    parser.add_argument('--base_model', type=str, default='InstaDeepAI/nucleotide-transformer-v2-500m-multi-species')
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU to use (default:0)')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--prediction_path', type=str, help='Location to print predictions', default='./')
    
    args = parser.parse_args()
    #if args.reference_model is None:
    #    args.reference_model = args.tokenizer
    logging.info(args)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, model_max_length=1024, trust_remote_code=True)
    device = torch.device(f"cuda:{args.gpu_id}") if torch.cuda.is_available() and args.gpu_id >= 0 else torch.device("cpu")

    eval_data = pd.read_csv('03.TF_binding_table_Phchr4_2.csv', sep=',')
    labels = eval_data[f'{args.tf_id}'].tolist()
    data = eval_data['upstream_CDS_1000bp'].tolist()

    eval_data = Dataset.from_dict({'sequence': data, 'labels': labels})
    eval_data = process_data(eval_data, tokenizer, shuffle=False)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision('high')

    metric = evaluate.load('glue', 'mrpc')

    base_model = AutoModelForSequenceClassification.from_pretrained(args.base_model, num_labels=2, trust_remote_code=True)
    ft_model = PeftModel.from_pretrained(base_model, args.model).merge_and_unload(safe_merge=True).to(device)
    result, out_probs, etalons = eval_model(ft_model, eval_data, args.batch_size, metric)
    for k,v in result.items():
        print(f'{k} {v:.4f}')

if __name__=='__main__':
    main()
