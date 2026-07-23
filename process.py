import os, sys, argparse, gzip
import numpy as np
import torch
from tqdm.auto import tqdm
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class Processor(object):

    def __init__(self, model_name, device_id, tokenizer_name=None):
        device_count = torch.cuda.device_count()
        if device_count != 0 and device_id >= 0:
            if device_id >= device_count:
                device_id = random.randint(1, device_count) - 1
            device = torch.device(f'cuda:{device_id}')
        else:
            device =  torch.device('cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name if tokenizer_name is None else tokenizer_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, output_hidden_states=True).to(device)

    def get_representations(self, data, batch_size, target_layer=-1, only_cls=True, to_skip=None):
        representations = []
        #num_lines = len([None for _ in open(file_name)])
        #with open(file_name) as fi:
        #for i,l in tqdm(enumerate(fi), total=num_lines):
        sequences = [seq for meta_data, seq in data]
        num_batches = int(np.ceil(len(data) / batch_size))
        for b in tqdm(range(num_batches), total=num_batches):
        #for i,(meta_data, sequence) in tqdm(enumerate(data), total=len(data)):
            tokens = self.tokenizer(sequences[b*batch_size:(b+1)*batch_size], padding=True, return_tensors='pt')

            with torch.inference_mode():
                outputs = self.model(**tokens.to(self.model.device))
                length = outputs.hidden_states[target_layer].shape[1]
                h = outputs.hidden_states[target_layer][:, 0:(1 if only_cls else length)].cpu().numpy()
                representations.append(h)
        return np.vstack(representations)

def process_fasta(fn, omit_star=True, deduplicate_seqs=False):
    to_ret = {}
    seqs_seen = set()
    metadata, seq = None, ''
    with gzip.open(fn, 'rt') if fn.endswith('.gz') else open(fn) as fi:
        for l in fi:
            l = l.strip()
            if l=='torol':
               break
            if len(l) > 0 and l[0]=='>':
                if metadata:
                    if metadata in to_ret:
                        print(f'WARNING: duplicate {seq} vs {to_ret[metadata]}')
                    if not deduplicate_seqs or seq not in seqs_seen:
                        to_ret[metadata] = seq
                        seqs_seen.add(seq)
                metadata = l[1:]
                seq = ''
            else:
                seq += l.replace('*', '') if omit_star else l
    
    if metadata and len(seq) > 0:
        if metadata in to_ret:
            print(f'WARNING: duplicate {seq} vs {to_ret[metadata]}')
        if not deduplicate_seqs or seq not in seqs_seen:
            to_ret[metadata] = seq
    return [(k, to_ret[k]) for k in sorted(to_ret)]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('input_file')
    parser.add_argument('--model_layers', type=int, default=6)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--layer', type=int)
    parser.add_argument('--batch_size', type=int, default=256)

    args = parser.parse_args()
    if args.layer is None:
        args.layer = args.model_layers
    assert args.layer <= args.model_layers
    print(args)

    #out_data_file = f'{args.input_file}.data'
    #if os.path.exists(out_data_file):
    #    print(f'{out_data_file} already exists')
    #else:
    data = process_fasta(args.input_file)
    #with open(out_data_file, 'w') as fo:
    #    for meta_data, seq in data:
    #        fo.write(f'{meta_data.replace("\t", " ")}\t{seq}\n')
    #print(f'{out_data_file} created')

    only_cls = True
    out_repr_file = f'representations/{os.path.basename(args.input_file)}_{args.model_layers}_{args.layer}_{only_cls}.npy'
    if not os.path.exists(out_repr_file):
        model_sizes = {6:'8M', 12:'35M', 30:'150M', 33:'650M', 36:'3B', 48:'15B'}

        if args.model_layers not in model_sizes:
            print("Model size unavailable")
            sys.exit(2)

        p = Processor(f'facebook/esm2_t{args.model_layers}_{model_sizes[args.model_layers]}_UR50D', args.gpu)
        representations = p.get_representations(data, args.batch_size, target_layer=args.layer, only_cls=only_cls).squeeze()
        print(representations.shape)

        np.save(out_repr_file, representations)

