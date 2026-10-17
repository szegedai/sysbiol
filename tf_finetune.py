import os, re, sys, copy, gzip, glob
import argparse
import torch
import transformers
import numpy as np
import pandas as pd
import random

from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from tqdm.auto import tqdm
from datasets import Dataset
from process import process_fasta

from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType
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

def eval_model(model, eval_data, batch_size, progress_bar=None, tokenizer=None):
    model.eval()
    out_probs, preds, etalons = [], [], []
    for batch_id in tqdm(range(int(np.ceil(len(eval_data) / batch_size)))):
        to_index = min(len(eval_data), (batch_id+1)*batch_size)
        batch_seqs = eval_data.select(range(batch_id*batch_size, to_index))
        batch = pad_batch(eval_data.select(range(batch_id*batch_size, to_index)), model.device)
        with torch.no_grad():
            outputs = model(**batch)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        out_probs.extend(probs[:,1].tolist())
        etalons.extend(batch['labels'].tolist())
        predictions = torch.argmax(probs, dim=-1)
        preds.extend(predictions.tolist())
        if progress_bar is not None:
            progress_bar.update(1)
    metrics = {'f1': f1_score(etalons, preds), 'roc': roc_auc_score(etalons, out_probs), 'accuracy': accuracy_score(etalons, preds), 'confusion_matrix': confusion_matrix(etalons, preds), 'classification_report': classification_report(etalons, preds)}
    return metrics, out_probs, etalons

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

def get_optimizer_and_scheduler(m, training_steps, lr=2e-5, warmup_ratio=0.0, weight_decay=0.01):
    optimizer = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=weight_decay)
    #optimizer = torch.optim.SGD(m.parameters(), lr=lr)

    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=int(warmup_ratio * training_steps),
        num_training_steps=training_steps)
    return optimizer, scheduler


def train(experiment_id, model, num_epochs, training_data, eval_data, batch_size, grad_accum,
          lr=2e-5, warmup_rate=0, shuffle_seed=42, log_predictions=False, clip=0.0, weight_decay=0.01, weight_log=False, use_amp=False, dtype=None):

    amp_settings = {"device_type":model.device.type,
                    "enabled":use_amp and model.device.type!='cpu',
                    "dtype": dtype}

    ref_model = copy.deepcopy(model)
    ref_model.to('cpu')

    losses = [0.0]
   
    num_training_steps = num_epochs * int(np.ceil(len(training_data) / (batch_size * grad_accum)))
    num_eval_steps = num_epochs * (0 if eval_data is None else int(np.ceil(len(eval_data) / batch_size)))

    full_macrobatches_per_epoch = len(training_data) // (batch_size * grad_accum)
    minibatches_per_epoch = int(np.ceil(len(training_data) / batch_size))
    last_minibatch_number = int(np.ceil((len(training_data) - full_macrobatches_per_epoch * (batch_size * grad_accum)) / batch_size))
    
    optimizer, lr_scheduler = get_optimizer_and_scheduler(model, num_training_steps, lr, warmup_rate, weight_decay)
    progress_bar = tqdm(range(num_training_steps + num_eval_steps), file=sys.stderr)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    np.random.seed(shuffle_seed)
    
    for epoch in range(num_epochs):
        model.train()

        random_samples = np.random.permutation(range(len(training_data)))
        for bi, from_idx in enumerate(range(0, len(training_data), batch_size)):
            to_index = min(len(training_data), from_idx+batch_size)
            batch = pad_batch(training_data.select(random_samples[from_idx:to_index]), model.device)

            #if epoch==0 and bi==0:
            #    if len(batch['input_ids'].shape)==2:
            #        logging.info(batch['input_ids'][0:3,0:8])
            
            #for k,v in batch.items():
            #    logging.info((k,v))
            #logging.info((model.num_heads, model.num_labels))
            with torch.autocast(**amp_settings):
                outputs = model(**batch)
    
            normalizer = grad_accum if bi < grad_accum * full_macrobatches_per_epoch else last_minibatch_number

            loss = outputs.loss / normalizer
            losses[-1] += loss.item()
            scaler.scale(loss).backward()

            if ((bi+1) % grad_accum == 0) or bi == minibatches_per_epoch - 1:
                scaler.unscale_(optimizer)
                if clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                scaler.step(optimizer) #optimizer.step()
                scaler.update()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                losses.append(0.0)

        if weight_log:
            for param_name, params in model.named_parameters():
                delta_weights = params.detach().cpu() - ref_model.get_parameter(param_name)
                logging.info(f"WEIGHT@{epoch}\t{experiment_id}\t{param_name}\t{torch.linalg.norm(delta_weights).item()}\t{np.prod(params.size())}")

        if eval_data is not None:
            performances, out_probs, etalons = eval_model(model, eval_data, batch_size, progress_bar)
            for performance, value in performances.items():
                logging.info(f"{epoch} {performance} {value:.4f}")

    return losses

def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    torch.cuda.manual_seed_all(s)

def main():
    parser = argparse.ArgumentParser(description='Finetuning experiments.')

    parser.add_argument('--tf_id')
    parser.add_argument('--training_folder', default='./CopciAB')
    parser.add_argument('--model', type=str, default='InstaDeepAI/nucleotide-transformer-v2-500m-multi-species')
    #InstaDeepAI/nucleotide-transformer-2.5b-multi-species
    #parser.add_argument('--model', type=str, default='InstaDeepAI/NTv3_650M_pre')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU to use (default:0)')
    parser.add_argument('--train_samples', type=int, default=-1)
    parser.add_argument('--epochs', type=int, default=4, help='Epochs to run (default:3)')
    parser.add_argument('--num_experiments', type=int, default=3, help='Epochs to run (default:3)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (default:32)')
    parser.add_argument('--grad_accum', type=int, default=1, help='Batch size (default:1)')
    parser.add_argument('--tunable', type=int, default=-1, help='Tunable layers (default:-1)')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate to applay (default:2e-5)')
    parser.add_argument('--warmup_ratio', type=float, default=0, help='Warmup ratio (default:0)')
    parser.add_argument('--start_seed', type=int, default=0, help='Finetuning seed (default:0)')
    parser.add_argument('--num_heads', type=int, default=1, help='Number of classifier heads (default:1)')
    parser.add_argument('--smoothing', type=float, default=0.0, help='Smoothing factor (default:0.0)')
    parser.add_argument('--grad_clipping', type=float, default=0.0, help='Clip gradients at the given threshold for >0 values (default:0.0)')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay to use (default:0.01)')
    parser.add_argument('--lora_r', type=int, default=16)

    parser.add_argument('--amp', dest='amp', action='store_true')
    parser.add_argument('--not-amp', dest='amp', action='store_false')
    parser.set_defaults(amp=False)

    parser.add_argument('--verbose_loss', dest='log_losses', action='store_true')
    parser.add_argument('--silent_loss', dest='log_losses', action='store_false')
    parser.set_defaults(log_losses=False)

    parser.add_argument('--verbose_predictions', dest='log_predictions', action='store_true')
    parser.add_argument('--silent_predictions', dest='log_predictions', action='store_false')
    parser.set_defaults(log_pedictions=False)

    parser.add_argument('--trainable_head', dest='trainable_head', action='store_true')
    parser.add_argument('--no_trainable_head', dest='trainable_head', action='store_false')
    parser.set_defaults(trainable_head=True)
    
    parser.add_argument('--log_weights', dest='weight_log', action='store_true')
    parser.add_argument('--no_log_weights', dest='weight_log', action='store_false')
    parser.set_defaults(weight_log=False)
    
    parser.add_argument('--multihead', dest='mh', action='store_true')
    parser.add_argument('--no_multihead', dest='mh', action='store_false')
    parser.set_defaults(mh=False)
    
    parser.add_argument('--save_model', default='./tf_models', type=str, help='Location to save file')
    parser.add_argument('--eval_frac', default=0.2, type=float)   
 
    args = parser.parse_args()
    args.mh = args.mh or args.num_heads > 1
    logging.info(args)

    tokenizer = AutoTokenizer.from_pretrained(args.model, model_max_length=1024, trust_remote_code=True)
    device = torch.device(f"cuda:{args.gpu_id}") if torch.cuda.is_available() and args.gpu_id >= 0 else torch.device("cpu")

    data, labels = [], []
    for f in glob.glob(f'{args.training_folder}/*.fasta*'):
        if args.tf_id is None or args.tf_id in f:
            d = process_fasta(f, deduplicate_seqs=False)
            logging.info(f'{f} {len(d)}')
            data.extend([x[1] for x in d])
            label = 1 if 'positive' in f else 0
            labels.extend([label for _ in range(len(d))])
    
    logging.info(f'{len(data)} {len(labels)} {np.sum(labels)}')

    train_data = Dataset.from_dict({'sequence': data, 'labels': labels})
    train_data = process_data(train_data, tokenizer, samples_to_use=args.train_samples, shuffle=True)

    cutoff_value = int(np.round(len(train_data) * args.eval_frac))
    eval_data = train_data.select(range(cutoff_value))
    train_data = train_data.select(range(cutoff_value, len(train_data)))

    eval2_data = pd.read_csv('03.TF_binding_table_Phchr4_2.csv.gz', sep=',')
    data = eval2_data['upstream_CDS_1000bp'].tolist()
    eval2_data = Dataset.from_dict({'sequence': data, 'labels': eval2_data[f'{args.tf_id}'].tolist()})
    eval2_data= process_data(eval2_data, tokenizer, shuffle=False)

    logging.info(f'{len(train_data)} {len(eval_data)} {len(eval2_data)}')

    torch.set_float32_matmul_precision('high')
    for run in range(args.num_experiments):
        model_path = f"{args.save_model}/{args.tf_id}_{args.epochs}_{args.start_seed+run}_{args.num_heads}_{args.smoothing}_{args.trainable_head}_{args.lora_r}_{args.model.replace('/', '_')}"
        set_seed((seed:=args.start_seed + run))
        if os.path.exists(model_path) is False:
            logging.info(f"{args.save_model}\t{args.tf_id}_{args.epochs}_{args.start_seed+run}_{args.num_heads}_{args.smoothing}_{args.trainable_head}_{args.model.replace('/', '_')}")
            model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2, trust_remote_code=True).to(device)

            if args.lora_r > 0:
                peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, inference_mode=False, r=args.lora_r, lora_alpha=args.lora_r, lora_dropout=0.1, target_modules='all-linear', bias="all")
                model = get_peft_model(model, peft_config)
                model.print_trainable_parameters()

            layers_to_train = ['classifier', 'pooler',] + [f'layer.{l}' for l in range(model.config.num_hidden_layers - args.tunable, model.config.num_hidden_layers)]
            logging.info(layers_to_train)
    
            if args.tunable > -1:
                for p in model.named_parameters():
                    p[1].requires_grad = np.any([x in p[0] for x in layers_to_train])

            results, losses = train(run, model, args.epochs, train_data, eval_data, args.batch_size, args.grad_accum,
                                    lr=args.lr, warmup_rate=args.warmup_ratio, log_predictions=args.log_predictions,
                                    clip=args.grad_clipping, weight_decay=args.weight_decay, weight_log=args.weight_log, use_amp=args.amp)
            if args.save_model:
                model.save_pretrained(model_path)
            logging.info(f'Finetuning finished and written to {model_path}')
        else:
            if os.path.exists(f'{model_path}/adapter_config.json'):
                base_model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2, trust_remote_code=True)
                model = PeftModel.from_pretrained(base_model, model_path).merge_and_unload(safe_merge=True).to(device)
            else:
                model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2, trust_remote_code=True).to(device)
        
        for ed_name, ed in zip('in out'.split(), [eval_data, eval2_data]):
            performances2, out_probs, etalons = eval_model(model, ed, 1024)
            if ed_name=='out':
                with open(f'{model_path.replace("/", "_")[2:]}_{args.tf_id}.out', 'w') as fo:
                    fo.write(f'{args.tf_id}\n')
                    for i,(op, et) in enumerate(zip(out_probs, etalons)):
                        fo.write(f'{op:.5f}\n')
            with open(f'{args.tf_id}_{seed}_{ed_name}Domain.log', 'w') as fo:
                for performance, value in performances2.items():
                    fo.write(f"{ed_name}-data\t{performance}\t{value}\n")

if __name__=='__main__':
    main()
