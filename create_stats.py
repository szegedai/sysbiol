import os, re, sys, gzip, argparse, numpy as np
from collections import Counter
from process import process_fasta

def main():
    parser = argparse.ArgumentParser(description='Finetuning experiments.')
    parser.add_argument('genom_file')
    parser.add_argument('--feromon_file', default='data_trial/413609479704199_wo_noise.fasta.gz')
    args = parser.parse_args()
    
    genom_data = process_fasta(args.genom_file) #read_data(f'{args.genom_file}.data')
    feromon_data = process_fasta(args.feromon_file) #read_data(f'{args.feromon_file}.data')
    
    prefix = os.path.basename(args.genom_file).split('_')[0] # this prefix is to be discarded from the training samples

    to_keep = [prefix not in fd[0] for fd in feromon_data]
    training_feromons = [fd[1] for fd,keep in zip(feromon_data, to_keep) if keep]

    genom_candidates = np.array([re.match('.*C[AVILT]{2}[A-Z]$', gd[1]) is not None for gd in genom_data])
    neg_genom_data = [gd[1] for gd, candidate in zip(genom_data, genom_candidates) if candidate==False]
    test_genom_data = [gd for gd, candidate in zip(genom_data, genom_candidates) if candidate==True]
    total_len = sum([len(seq) for seq in neg_genom_data + training_feromons])
    training_size = len(training_feromons) + len(neg_genom_data)
    print(f'{prefix};{(total_len / training_size):.3f};{len(genom_candidates)};{len(neg_genom_data)};{sum(genom_candidates)};{sum(to_keep)}') 

if __name__=='__main__':
    main()
