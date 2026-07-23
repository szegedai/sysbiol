import os, re, sys
from tqdm.auto import tqdm
import spams
import numpy as np
import argparse
import scipy
import itertools

from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

from process import process_fasta, Processor

def dense_pmi(incidence_mtx, labels, norm_pmi=True, nonneg_pmi=False):
    coocc = np.zeros((incidence_mtx.shape[1], 2))
    for label in range(2):
        aggregate = incidence_mtx[labels==label].sum(axis=0)
        coocc[:, label] = aggregate

    total_observations = coocc.sum()
    row_marginals = np.sum(coocc, axis=1) / total_observations
    col_marginals = np.sum(coocc, axis=0) / total_observations
    joint_probs = coocc / total_observations
    outerP = np.outer(row_marginals, col_marginals)

    # masked operations allow for avoiding numeric issues, e.g dividing by 0
    pmi = np.ma.log(np.ma.divide(outerP, joint_probs).filled(1)).filled(0)
    if norm_pmi:
        pmi = np.ma.divide(pmi, np.ma.log(joint_probs).filled(0)).filled(0)
    if nonneg_pmi:
        pmi[pmi<0] = 0
    return pmi

def transform_atoms(M, weight=False, use_singletons=True, use_pairs=True):
    assert use_singletons or use_pairs

    if weight==False:
        M.data = np.ones_like(M.data)

    num_samples, num_atoms = M.shape

    if use_singletons and not use_pairs:
        filter_fun = lambda x: x[0][0]==x[1][0]
    elif use_singletons:
        filter_fun = lambda x: x[0][0]>=x[1][0]
    else:
        filter_fun = lambda x: x[0][0]>x[1][0]

    data, indices, indptrs = [], [], [0]
    for sid in range(num_samples):

        z = list(zip(M[sid].indices, M[sid].data))
        for (ind1, val1),(ind2, val2) in filter(filter_fun, itertools.product(z,z)):
            data.append(val1 * val2)
            indices.append(ind2 * num_atoms + ind1)
        indptrs.append(len(data))
    return scipy.sparse.csr_matrix((data, indices, indptrs), shape=(num_samples, num_atoms **2))

def print_preds(data, scores, labels, out_dir, out_file, threshold=None, top_k=None, verbose=True):
    sorted_indices = np.argsort(scores)
    preds, golds = [], []
    correct, total = 0, 0
    to_print = False
    with open(f'{out_dir}/{out_file}', 'w') as fo, open(f'{out_dir}/reduced/{out_file}', 'w') as rfo:
        for o, ind in enumerate(sorted_indices[::-1]):
            golds.append(labels[ind])
            correct += labels[ind]
            total += 1
            if ((top_k and total <= top_k) or (threshold is not None and scores[ind] > threshold)):
                rfo.write(f'{";".join(data[ind])};{labels[ind]};{scores[ind]:.4f};{(correct/total):.3f};{o+1}\n')
                preds.append(True)
                if verbose:
                    print(f'{";".join(data[ind])}\t{labels[ind]}\t{scores[ind]:.4f}\t{(correct/total):.4f}\t{o+1}')
                    to_print = True
            else:
                preds.append(False)
            fo.write(f'{";".join(data[ind])};{labels[ind]};{scores[ind]:.4f};{(correct/total):.3f};{o+1}\n')
    f1 = f1_score(golds, preds) if 0 < sum(golds) < len(golds) else 0.0
    acc = accuracy_score(golds, preds)
    return f1, acc, sum(preds), sorted_indices

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Performs dictionary learning of hidden representations.')
    parser.add_argument('genom_file')
    parser.add_argument('--feromon_file', default='data_trial/413609479704199_wo_noise.fasta.gz')
    parser.add_argument('--model_layers', default=6, type=int)
    parser.add_argument('--layer', default=1, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--batch_size', default=8192, type=int)
    parser.add_argument('--transformer', type=str)
    parser.add_argument('--out_dir', default='/srv/data/berend/szbk/predictions/', type=str)
    parser.add_argument('--num_dict_atoms', type=int, default=1024, help='The number of semantic atoms in the learned dictionary matrix (default: 1024)')
    
    args = parser.parse_args()

    dl_params = {'K': args.num_dict_atoms, 'lambda1': 0.05, 'numThreads': 8, 'iter': 500, 'batchsize': 512, 'posAlpha': True, 'verbose': False}
    l_param = {x:dl_params[x] for x in ['L','lambda1','lambda2','mode','pos','ols','numThreads','length_path','verbose'] if x in dl_params}
    l_param['pos'] = True
    
    genom_location = f'{args.genom_file.replace("data/", "representations/")}_{args.model_layers}_{args.layer}_True.npy'
    feromon_location = f'{args.feromon_file.replace("data/", "representations/")}_{args.model_layers}_{args.layer}_True.npy'

    genom_data = process_fasta(args.genom_file)
    feromon_data = process_fasta(args.feromon_file)

    prefix = os.path.basename(args.genom_file).split('_')[0] # this prefix is to be discarded from the training samples

    to_keep = [prefix not in fd[0] for fd in feromon_data]
    training_feromons = [fd for fd,keep in zip(feromon_data, to_keep) if keep]

    genom_candidates = np.array([re.match('.*C[AVILT]{2}[A-Z]$', gd[1]) is not None for gd in genom_data])
    neg_genom_data = [gd for gd, candidate in zip(genom_data, genom_candidates) if candidate==False]
    if len(neg_genom_data) > 3000000:
        np.random.seed(42)
        np.random.shuffle(neg_genom_data)
        neg_genom_data = neg_genom_data[0:3000000]
    test_genom_data = [gd for gd, candidate in zip(genom_data, genom_candidates) if candidate==True]

    vectorizer = CountVectorizer(ngram_range=(args.layer,args.layer), analyzer='char')
    decorate = lambda L: [f'^{seq}$' for comment, seq in L]

    training_X_vec = vectorizer.fit_transform(decorate(neg_genom_data + training_feromons))
    y = np.array([0 for _ in neg_genom_data] + [1 for _ in training_feromons])
    test_vec = vectorizer.transform(decorate(test_genom_data))
    
    known_gold_seqs = set([fd[1] for fd in feromon_data if prefix in fd[0]])
    labels = np.array([genom_subseq in known_gold_seqs for _, genom_subseq in test_genom_data])
    #print(len(test_genom_data), len(neg_genom_data), len(feromon_data), len(training_feromons))
    print(args, len(neg_genom_data))

    nb_model = MultinomialNB().fit(training_X_vec, y)
    y_pred_nb = nb_model.predict_proba(test_vec)[:,1]
    f1, acc, pos_preds, _ = print_preds(test_genom_data, y_pred_nb, labels, args.out_dir, f'C{args.layer}_NB_{prefix}.csv', threshold=0.5, verbose=False)
    print(f"{prefix}\tNB\t{args.layer}\t{pos_preds}\t{f1:.4f}\t{acc:.4f}")
   
    model_sizes = {6:'8M', 12:'35M', 30:'150M', 33:'650M', 36:'3B', 48:'15B'} 
    tokenizer_name = f'facebook/esm2_t{args.model_layers}_{model_sizes[args.model_layers]}_UR50D'
    p = Processor(tokenizer_name if args.transformer is None else args.transformer, args.gpu, tokenizer_name)
    feromon_X = p.get_representations(feromon_data, args.batch_size, target_layer=args.layer, only_cls=True).squeeze()[to_keep]
    negative_X =  p.get_representations(neg_genom_data, args.batch_size, target_layer=args.layer, only_cls=True).squeeze()
    test_genom_X =  p.get_representations(test_genom_data, args.batch_size, target_layer=args.layer, only_cls=True).squeeze()

    training_X = np.vstack((negative_X, feromon_X))

    np.random.seed(42)
    permutation = np.random.permutation(list(range(len(y)))) 
    training_X = np.array([training_X[row_id,:] for row_id in permutation])
    y = np.array([y[p] for p in permutation])

    lr_model = LogisticRegression(random_state=0).fit(training_X, y)
    y_pred_lr = lr_model.predict_proba(test_genom_X)[:,1]
    f1, acc, pos_preds, _ = print_preds(test_genom_data, y_pred_lr, labels, args.out_dir, f'L{args.layer}_LR_{prefix}.csv', threshold=0.5, verbose=False)
    print(f"{prefix}\tLR\t{args.layer}\t{pos_preds}\t{f1:.4f}\t{acc:.4f}")

    dict_file = f'dictionaries/{prefix}_L{args.layer}_K{args.num_dict_atoms}.dict'
    if os.path.exists(f'{dict_file}.npy'):
        D = np.load(f'{dict_file}.npy')
    else:
        E = normalize(training_X).T
        D = spams.trainDL(E, **dl_params)
        #np.save(dict_file, D)
        train_alphas = spams.lasso(E, D=D, **l_param).T
        test_alphas = spams.lasso(normalize(test_genom_X).T, D=D, **l_param).T
        train_alphas.data = np.nan_to_num(train_alphas.data) 
        test_alphas.data = np.nan_to_num(test_alphas.data) 
        #scipy.sparse.save_npz(f'alphas/{prefix}_L{args.layer}_{args.num_dict_atoms}.train', train_alphas)
        #scipy.sparse.save_npz(f'alphas/{prefix}_L{args.layer}_{args.num_dict_atoms}.test', test_alphas)

    #train_alphas = scipy.sparse.load_npz(f'alphas/{prefix}_L{args.layer}_{args.num_dict_atoms}.train.npz')
    #test_alphas = scipy.sparse.load_npz(f'alphas/{prefix}_L{args.layer}_{args.num_dict_atoms}.test.npz')
    lr_model = LogisticRegression(random_state=0).fit(train_alphas, y)
    y_pred_lr = lr_model.predict_proba(test_alphas)[:,1]
    f1, acc, pos_preds, _ = print_preds(test_genom_data, y_pred_lr, labels, args.out_dir, f'L{args.layer}K{args.num_dict_atoms}_sLR_{prefix}.csv', threshold=0.5, verbose=False)
    print(f"{prefix}\tsLR\t{args.layer}\t{pos_preds}\t{f1:.4f}\t{acc:.4f}")
    
    train_alphas.data = np.ones_like(train_alphas.data)
    test_alphas.data = np.ones_like(test_alphas.data)
    
    lr_model = LogisticRegression(random_state=0).fit(train_alphas, y)
    y_pred_lr = lr_model.predict_proba(test_alphas)[:,1]
    f1, acc, pos_preds, _ = print_preds(test_genom_data, y_pred_lr, labels, args.out_dir, f'L{args.layer}K{args.num_dict_atoms}_sbLR_{prefix}.csv', threshold=0.5, verbose=False)
    print(f"{prefix}\tsbLR\t{args.layer}\t{pos_preds}\t{f1:.4f}\t{acc:.4f}")
    
    nb_model = MultinomialNB().fit(train_alphas, y)
    y_pred_nb = nb_model.predict_proba(test_alphas)[:,1]
    f1, acc, pos_preds, _ = print_preds(test_genom_data, y_pred_nb, labels, args.out_dir, f'L{args.layer}K{args.num_dict_atoms}_sNB_{prefix}.csv', threshold=0.5, verbose=False)
    print(f"{prefix}\tsNB\t{args.layer}\t{pos_preds}\t{f1:.4f}\t{acc:.4f}")

    '''
    PMI = dense_pmi(train_alphas, y)
    scores = test_alphas @ PMI
    y_pred_pmi = scores[:,1] - scores[:,0]
    f1, acc, pos_preds, _ = print_preds(test_genom_data, y_pred_pmi, labels, f'predictions/L{args.layer}K{args.num_dict_atoms}_pmi_{prefix}.csv', threshold=0.0, verbose=False)
    print(f"{prefix}\tPMI\t{args.layer}\t{pos_preds}\t{f1:.4f}\t{acc:.4f}")
    #print(np.max(PMI, axis=1))
    '''
    train_alphas2 = transform_atoms(train_alphas, weight=False, use_singletons=True, use_pairs=True)
    test_alphas2 = transform_atoms(test_alphas, weight=False, use_singletons=True, use_pairs=True)
    
    '''
    PMI2 = dense_pmi(train_alphas2, y)
    scores = test_alphas2 @ PMI2
    y_pred_pmi = scores[:,1] - scores[:,0]
    f1, acc, pos_preds, _ = print_preds(test_genom_data, y_pred_pmi, labels, args.out_dir, f'L{args.layer}K{args.num_dict_atoms}_pmi2_{prefix}.csv', threshold=0.0, verbose=False)
    print(f"{prefix}\tPMI2\t{args.layer}\t{pos_preds}\t{f1:.4f}\t{acc:.4f}")
    '''

    lr_model = LogisticRegression(random_state=0).fit(train_alphas2, y)
    y_pred_lr = lr_model.predict_proba(test_alphas2)[:,1]
    f1, acc, pos_preds, _ = print_preds(test_genom_data, y_pred_lr, labels, args.out_dir, f'L{args.layer}K{args.num_dict_atoms}_LR2_{prefix}.csv', threshold=0.5, verbose=False)
    print(f"{prefix}\tsLR2\t{args.layer}\t{pos_preds}\t{f1:.4f}\t{acc:.4f}")

    nb_model = MultinomialNB().fit(train_alphas2, y)
    y_pred_nb = nb_model.predict_proba(test_alphas2)[:,1]
    f1, acc, pos_preds, _ = print_preds(test_genom_data, y_pred_nb, labels, args.out_dir, f'L{args.layer}K{args.num_dict_atoms}_sNB2_{prefix}.csv', threshold=0.5, verbose=False)
    print(f"{prefix}\tsNB2\t{args.layer}\t{pos_preds}\t{f1:.4f}\t{acc:.4f}")

    for method in ['max']: #'mean max min'.split():
        if method=='mean':
            fun = np.mean
        elif method=='max':
            fun = np.max
        elif method=='min':
            fun = np.min

        chunk_size = 128
        num_chunks = test_genom_X.shape[0] // chunk_size + 1
        pos_scores = np.concat([fun(cosine_similarity(test_genom_X[c*chunk_size:(c+1)*chunk_size], feromon_X), axis=1) for c in range(num_chunks)])
        neg_scores = np.concat([fun(cosine_similarity(test_genom_X[c*chunk_size:(c+1)*chunk_size], negative_X), axis=1) for c in range(num_chunks)])

        preds = pos_scores > neg_scores
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds)
        
        f1, acc, pos_preds, sorted_indices = print_preds(test_genom_data, pos_scores-neg_scores, labels, args.out_dir, f'L{args.layer}_supDist_{prefix}.csv', threshold=0.0, verbose=False)
        print(f"{prefix}\tsDIST\t{args.layer}\t{pos_preds}\t{f1:.4f}\t{acc:.4f}")

        #auc = roc_auc_score(labels, pos_scores)
        #ap = average_precision_score(labels, pos_scores)
        
        f1, acc, pos_preds, sorted_indices = print_preds(test_genom_data, pos_scores, labels, args.out_dir, f'L{args.layer}_unsupDist_{prefix}.csv', top_k=50, verbose=False)
        print(f"{prefix}\tuDIST\t{args.layer}\t{pos_preds}\t{f1:.4f}\t{acc:.4f}")
        #top_k_f1 = []
        #for k in [5, 10, 20, 50]:
        #    pred_labels = [s >= pos_scores[sorted_indices[-k]] for s in pos_scores]
        #    top_k_f1.append(f'{f1_score(labels, pred_labels):.4f}')
        #top_ks_str = '_'.join(top_k_accuracies)
        #print(f'{args.model_layers}\t{args.layer}\t{method}\t{acc:.4f}\t{f1:.4f}\t{auc:.4f}\t{ap:.4f}\t{"\t".join(top_k_f1)}\t{np.mean(pos_scores):.4f}\t{np.median(pos_scores):.4f}\t{np.std(pos_scores):.4f}\t{np.min(pos_scores):.4f}')
        #print(f'{args.model_layers}\t{args.layer}\t{method}\t{sum(preds)}\t{acc:.4f}\t{f1:.4f}\t{auc:.4f}\t{ap:.4f}\t{"\t".join(top_k_f1)}')
