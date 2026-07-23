import os, re, sys
import glob
import argparse
import pandas as pd
import numpy as np, seaborn as sns
import matplotlib.pyplot as plt

def merge_preds(combined_predictions):
    ds = pd.concat(combined_predictions, ignore_index=True)
    merged_preds = ds.groupby([0,1])[3].mean()
    return merged_preds[merged_preds > 0.5]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merges predictions based on multiple approaces.')
    parser.add_argument('--input_dir', help="The location of the prediction files", default='/srv/data/berend/szbk/predictions/')
    parser.add_argument('--pattern_to_keep', default='.*')
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--out_file', default='out.csv')

    args = parser.parse_args()

    files = os.listdir(args.input_dir)
    species = set([f.split('_')[-1].split('.')[0] for f in files if os.path.isfile(f'{args.input_dir}/{f}')])

    print(len(species), len(files))
    if args.pattern_to_keep:
        files = [f for f in files if re.match(args.pattern_to_keep, f)]
    print(len(files))

    species = list(sorted(species))
    methods = set(['wvote', 'uwvote'])
    predictions = {}
    freq_data = []
    for si, s in enumerate(species):
        #if si==5:break
        freq_data.append([s])
        num_preds = {}
        weighted_preds, unweighted_preds = [], []
        predictions[s] = {}
        for f in files:
            if f.endswith(f'{s}.csv'):
                *method, specimen = f.split('_')
                method = '_'.join(method)
                methods.add(method)
                data = pd.read_csv(f'{args.input_dir}/{f}', sep=';', header=None)
                predictions[s][method] = set(data[data[3]>0.5][1])
                weighted_preds.append(data)
                uw_data = data.copy()
                uw_data[3] = uw_data[3]>0.5
                unweighted_preds.append(uw_data)
                num_preds[method] = len(predictions[s][method])
        weighted_merged_pos_preds = merge_preds(weighted_preds)
        unweighted_merged_pos_preds = merge_preds(unweighted_preds)
        predictions[s]['wvote'] = set([ind[1] for ind in weighted_merged_pos_preds.index])
        predictions[s]['uwvote'] = set([ind[1] for ind in unweighted_merged_pos_preds.index])
        num_preds.update({'wvote': len(predictions[s]['wvote']), 'uwvote': len(predictions[s]['uwvote'])})
        freq_data[-1].extend([num_preds[m] for m in sorted(methods)])

        weighted_merged_pos_preds.sort_values(ascending=False).to_csv(f'./voted_predictions/wvote_{s}.csv', header=None, sep=';')
        unweighted_merged_pos_preds.sort_values(ascending=False).to_csv(f'./voted_predictions/uwvote_{s}.csv', header=None, sep=';')
        
        print(si+1, s, weighted_merged_pos_preds.shape, unweighted_merged_pos_preds.shape)

    freq_ds = pd.DataFrame(freq_data, columns=['species',]+[m for m in sorted(methods)])
    freq_ds.to_csv('method_predictions.csv', sep=';')

    methods = list(sorted(methods))
    global_intersections = {(m1,m2):0 for m1 in methods for m2 in methods}
    global_unions = {(m1,m2):0 for m1 in methods for m2 in methods}
    per_species_jaccard = {(m1,m2):[] for m1 in methods for m2 in methods}

    for species in sorted(predictions):
        for m1 in methods:
            p1_set = predictions[species][m1]
            for m2 in methods:
                p2_set = predictions[species][m2]
                intersection, union = p1_set & p2_set, p1_set | p2_set
                global_intersections[(m1,m2)] += len(intersection)
                global_unions[(m1,m2)] += len(union)
                jaccard = len(intersection) / len(union) if len(union) > 0 else 1
                per_species_jaccard[(m1,m2)].append(jaccard)

    m2i = {m:i for i,m in enumerate(methods)}
    macro_heatmap = np.zeros((len(methods), len(methods)))
    micro_heatmap = np.zeros((len(methods), len(methods)))
    m2i = {m:i for i,m in enumerate(methods)}
    for m1, i1 in m2i.items():
        for m2, i2 in m2i.items():
            macro_heatmap[i1,i2] = global_intersections[(m1,m2)] / global_unions[(m1,m2)]
            micro_heatmap[i1,i2] = np.mean(per_species_jaccard[(m1,m2)])

    fig, ax = plt.subplots(figsize=(15,13))
    sns.heatmap(macro_heatmap, xticklabels=methods, yticklabels=methods, ax=ax, annot=True, fmt='.1f', cmap="YlGnBu", square=True, annot_kws={"fontsize":8})
    plt.savefig('macro_heatmap.png')
    plt.close()

    fig, ax = plt.subplots(figsize=(16,13))
    sns.heatmap(micro_heatmap, xticklabels=methods, yticklabels=methods, ax=ax, annot=True, fmt='.1f', cmap="YlGnBu", square=True, annot_kws={"fontsize":8})
    plt.savefig('micro_heatmap.png')
    plt.close()

    '''scores = {}
    num_layers = 0
    for f in glob.glob(f"{args.pattern_prefix}*out"):
        num_layers += 1
        with open(f) as fi:
            for i,l in enumerate(fi):
                parts = l.strip().split(';')
                if i==0:
                    normalizer = float(parts[3])
                key = tuple(parts[0:3])
                if key not in scores:
                    scores[key] = 0.0
                scores[key] += float(parts[3]) / normalizer


    correct, total = 0,0
    with open(args.out_file, 'w') as fo:
        for i,(k,v) in enumerate(dict(sorted(scores.items(), key=lambda item: item[1], reverse=True)).items()):
            correct += 1 if k[-1]=='True' else 0
            total += 1
            if i<args.top_k:
                print(f"{';'.join(k)};{(v/num_layers):.4f};{(correct/total):.4f}")
            fo.write(f"{';'.join(k)};{(v/num_layers):.4f};{(correct/total):.4f}\n")
    '''        
