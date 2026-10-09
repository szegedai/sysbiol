import os, re, sys, copy, gzip
import argparse
import torch
import transformers
import numpy as np
import random

import evaluate
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from tqdm.auto import tqdm
from datasets import Dataset
from process import process_fasta

#from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType
from collections import Counter
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S',
                    stream=sys.stdout)

def get_optimizer_and_scheduler(m, training_steps, lr=2e-5, warmup_ratio=0.0, weight_decay=0.01):
    optimizer = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=weight_decay)
    #optimizer = torch.optim.SGD(m.parameters(), lr=lr)

    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=int(warmup_ratio * training_steps),
        num_training_steps=training_steps)
    return optimizer, scheduler


def pad_batch(batch, device):
    b = {}

    max_length = max([len(b) for b in batch['input_ids']])

    for f in batch.features:
        if f=='labels':
            b[f] = batch[f][:].to(device)
        else:
            b[f] = torch.vstack([torch.cat((b, torch.zeros(max_length - len(b), dtype=b.dtype))) for b in batch[f]]).to(device)
    return b

def train(experiment_id, model, metric, num_epochs, training_data, eval_data, batch_size, grad_accum,
          lr=2e-5, warmup_rate=0, shuffle_seed=42, log_predictions=False, clip=0.0, weight_decay=0.01, weight_log=False, use_amp=False, dtype=None):

    amp_settings = {"device_type":model.device.type,
                    "enabled":use_amp and model.device.type!='cpu',
                    "dtype": dtype}

    ref_model = copy.deepcopy(model)
    ref_model.to('cpu')

    metrics, losses = [], [0.0]
    probs_per_data = []
   
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

        if metric is not None and eval_data is not None:
            performances, preds, probs = eval_model(model, eval_data, batch_size, metric, progress_bar) #, tokenizer)
            probs_per_data.append(probs)
            for performance in performances:
                metrics.append(performance)
            if log_predictions and len(preds) > 0:
                logging.info(f"Predictions@{epoch}:{preds}")

    return metrics, losses, probs_per_data

def eval_model(model, eval_data, batch_size, metric, progress_bar):

    model.eval()
    model_outputs = []
    preds = [[] for _ in range(len(metric))]

    for bi in range(int(np.ceil(len(eval_data) / batch_size))):

        to_index = min(len(eval_data), (bi+1)*batch_size)
        batch = pad_batch(eval_data.select(range(bi*batch_size, to_index)), model.device)

        with torch.no_grad():
            outputs = model(**batch)
       
        if len(outputs.logits.shape)==1:
            outputs.logits = outputs.logits.unsqueeze(1)

        logits_shape = outputs.logits.shape
        # introduce a new dimension for the heads (i.e. split the outputs to num_heads x num_labels)
        out = outputs.logits.view(logits_shape[:-1] + (-1, model.num_labels))
        num_heads = out.shape[-2] # the penultimate dim is for the number of heads

        out = torch.nn.functional.softmax(out, dim=-1)
        predictions = torch.argmax(out, dim=-1) # batch size x [token num x] num head predictions tensor
        avg_output = torch.mean(out, dim=-2) # average across classification heads
        avg_predictions = torch.argmax(avg_output, dim=-1)

        model_outputs.append(out.detach().cpu())
        
        for h in range(num_heads):
            preds[h].extend([p.item() for p in predictions[...,h]])
            metric[h].add_batch(predictions=predictions[...,h], references=batch['labels'])

        if len(metric) > num_heads: # predictions averaged by classifier heads
            metric[-1].add_batch(predictions=avg_predictions, references=batch['labels'])
            preds[-1].extend(avg_predictions)

        if progress_bar is not None:
            progress_bar.update(1)
    return [m.compute() for m in metric], preds, model_outputs


def process_data(data, tokenizer, samples_to_use=-1, shuffle=True):
    
    def tokenize(examples):
        return tokenizer(examples['sentence'], padding=False, truncation=True)

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

def read_data(filename):
    return [l.strip().split('\t') for l in open(filename)]

def main():
    parser = argparse.ArgumentParser(description='Finetuning experiments.')

    parser.add_argument('genom_file')
    parser.add_argument('--feromon_file', default='data_trial/413609479704199_wo_noise.fasta.gz')
    parser.add_argument('--model', type=str, default='facebook/esm2_t6_8M_UR50D')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU to use (default:0)')
    parser.add_argument('--train_samples', type=int, default=-1)
    parser.add_argument('--epochs', type=int, default=3, help='Epochs to run (default:3)')
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
    
    parser.add_argument('--overwrite', dest='overwrite', action='store_true')
    parser.add_argument('--no_overwrite', dest='overwrite', action='store_false')
    parser.set_defaults(overwrite=False)
    
    parser.add_argument('--save_model', type=str, help='Location to save file')
    parser.add_argument('--prediction_path', type=str, help='Location to print predictions', default='./')
    
    args = parser.parse_args()
    #if args.reference_model is None:
    #    args.reference_model = args.tokenizer
    args.mh = args.mh or args.num_heads > 1
    logging.info(args)

    device = torch.device("cuda:{}".format(args.gpu_id)) if torch.cuda.is_available() and args.gpu_id >= 0 else torch.device("cpu")
    
    genom_data = process_fasta(args.genom_file, deduplicate_seqs=False) #read_data(f'{args.genom_file}.data')
    feromon_data = process_fasta(args.feromon_file, deduplicate_seqs=False) #read_data(f'{args.feromon_file}.data')
    
    prefix = os.path.basename(args.genom_file).split('_')[0] # this prefix is to be discarded from the training samples

    to_keep = [prefix not in fd[0] for fd in feromon_data]
    training_feromons = [fd[1] for fd,keep in zip(feromon_data, to_keep) if keep]

    genom_candidates = np.array([re.match('.*C[AVILT]{2}[A-Z]$', gd[1]) is not None for gd in genom_data])
    neg_genom_data = [gd[1] for gd, candidate in zip(genom_data, genom_candidates) if candidate==False][0:3000000] # we maximize the number of negative samples in 3e6
    test_genom_data = [gd for gd, candidate in zip(genom_data, genom_candidates) if candidate==True]
    
    y = np.array([0 for _ in neg_genom_data] + [1 for _ in training_feromons])

    known_gold_seqs = set([fd[1] for fd in feromon_data if prefix in fd[0]])
    test_labels = [1 if genom_subseq[1] in known_gold_seqs else 0 for genom_subseq in test_genom_data]

    tokenizer = AutoTokenizer.from_pretrained(args.model, model_max_length=1024)

    train_data = Dataset.from_dict({'sentence': neg_genom_data + training_feromons, 'labels': [0 for _ in neg_genom_data] + [1 for _ in training_feromons]})
    test_data = Dataset.from_dict({'sentence': [tgd[1] for tgd in test_genom_data], 'labels': test_labels})
    train_data= process_data(train_data, tokenizer, samples_to_use=args.train_samples, shuffle=True)
    test_data= process_data(test_data, tokenizer, samples_to_use=-1, shuffle=False)
    logging.info(f"PREFIX: {prefix} {len(neg_genom_data)} {len(training_feromons)} {len(test_genom_data)} {y.sum()} {sum(test_labels)}")

    for run in range(args.num_experiments):
        model_path = f"{args.save_model}/{prefix}_{args.epochs}_{args.start_seed+run}_{args.num_heads}_{args.smoothing}_{args.trainable_head}_{args.model.replace('/', '_')}"
        model_exists = os.path.exists(model_path)
        logging.info(f"SAVE\t{args.save_model}\t{prefix}_{args.epochs}_{args.start_seed+run}_{args.num_heads}_{args.smoothing}_{args.trainable_head}_{args.model.replace('/', '_')}")
        random.seed(args.start_seed + run)
        np.random.seed(args.start_seed + run)
        torch.manual_seed(args.start_seed + run)
        torch.cuda.manual_seed(args.start_seed + run)
        torch.cuda.manual_seed_all(args.start_seed + run)

        if args.overwrite or not model_exists:
            model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)
            num_epochs = args.epochs
        else:
            model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
            model.eval()
            num_epochs = 1 # we only want to evaluate an existing model in this case
            train_data = train_data.select(range(0))
          
        model = model.to(device)
        torch.set_float32_matmul_precision('high')

        #peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, inference_mode=False, r=16, lora_alpha=16, lora_dropout=0.1, bias="all")
        #model = get_peft_model(model, peft_config)
        #model.print_trainable_parameters()

        layers_to_train = ['classifier', 'pooler',] + [f'layer.{l}' for l in range(model.config.num_hidden_layers - args.tunable, model.config.num_hidden_layers)]
        logging.info(layers_to_train)
    
        if args.tunable > -1:
            for p in model.named_parameters():
                if np.any([x in p[0] for x in layers_to_train]):
                    p[1].requires_grad = True
                else:
                    p[1].requires_grad = False

        metric = [evaluate.load('glue', 'mrpc') for _ in range(args.num_heads + (0 if args.num_heads==1 else 1))]

        results, losses, eval_outs = train(run, model, metric, num_epochs, train_data, test_data, args.batch_size, args.grad_accum,
                                           lr=args.lr, warmup_rate=args.warmup_ratio, log_predictions=args.log_predictions,
                                           clip=args.grad_clipping, weight_decay=args.weight_decay, weight_log=args.weight_log, use_amp=args.amp)
        if args.save_model and args.overwrite:
            model.save_pretrained(model_path)
        logging.info('Finetuning finished')
        
        final_eval_outs = torch.concat(eval_outs[-1])[:,(head:=0),(positive_label_index:=1)].cpu().numpy()
        correct, total, total2 = 0, 0, 0
        pred_file = f'{args.prediction_path}/predictions/FT{run}_{prefix}.csv'
        reduced_pred_file = f'{args.prediction_path}/predictions/reduced/FT{run}_{prefix}.csv'
        with open(pred_file, 'w') as fo, open(reduced_pred_file, 'w') as rfo:
            for si_index, si in enumerate(np.argsort(final_eval_outs)[::-1]):
                if final_eval_outs[si] > 0.5:
                    correct += test_labels[si]
                    total += 1
                    rfo.write(f'{";".join(test_genom_data[si])};{test_labels[si]==1};{final_eval_outs[si]:.4f};{(correct / total):.4f};{si_index+1}\n')
                total2 += 1
                fo.write(f'{";".join(test_genom_data[si])};{test_labels[si]==1};{final_eval_outs[si]:.4f};{(correct / total2):.4f};{si_index+1}\n')

        if args.log_losses:
            for t in range(len(losses)//50):
                idx = (t+1)*50
                avg_loss = np.mean(losses[:idx])
                logging.info(f"LOSS\t{idx}\t{run}\t{avg_loss:.3f}\t{args.model}")
        logging.info(f"LOSS\t{len(losses)}\t{run}\t{np.mean(losses):.3f}\t{args.model}")

        for epoch, (res, outputs) in enumerate(zip(results, eval_outs)):
            #logging.info(f"outputs added {type(outputs)} {len(outputs)} {type(outputs[0])} {[o.shape for o in outputs]}")
            for metric_name, ms in res.items():
                to_log = ms if type(ms) == dict else {'': ms}
                for key, val in to_log.items():
                    logging.info(f"RESULTS-{metric_name}-{key.rstrip('-')}\t{args.batch_size}/{args.grad_accum}\t{args.start_seed+run}\t{epoch}\t{val:.4f}\t{args.model}")


if __name__=='__main__':
    main()
