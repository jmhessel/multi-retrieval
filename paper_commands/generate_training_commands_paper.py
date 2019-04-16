'''
This script generates the training commands for the results reported in the paper
'''
import collections
import numpy as np
import os

np.random.seed(1)
if not os.path.exists('paper_res'):
    os.makedirs('paper_res')
if not os.path.exists('paper_checkpoints'):
    os.makedirs('paper_checkpoints')

n_parallel = 4

all_commands = []

for batch_size in [11,21,31]:
    for dset in ['dii', 'diy', 'mscoco', 'rqa', 'sis', 'dii-r', 'wiki']:
        for alg in ['DC','TK','AP','NoStruct']:
            for k in [-1] if alg in ['DC', 'NoStruct'] else [-1, 2]:
                for neg_mining in ['negative_sample', 'hard_negative']:

                    # for wiki, just do limited settings
                    if dset == 'wiki' and (alg not in ['AP', 'NoStruct'] or k != -1 or neg_mining != 'hard_negative' or batch_size != 11): continue
                    
                    identifier_str = '{}+{}+{}+{}+{}'.format(dset, batch_size, alg, k, neg_mining)
                    output_f = 'paper_res/{}.pkl'.format(identifier_str)

                    cmd = 'python3 train_doc.py data/{}/docs.json --image_id2row data/{}/id2row.json --image_features data/{}/features.npy '.format(
                        dset, dset, dset)
                    if alg == 'NoStruct':
                        cmd += '--subsample_image 1 --subsample_text 1 '

                    if dset in ['rqa','diy','wiki']:
                        cmd += '--seq_len 50 '

                    cmd += '--gpu_memory_frac .45 --sim_mode {} --neg_mining {} --sim_mode_k {} '.format(alg if alg != 'NoStruct' else 'DC', neg_mining, k)
                    cmd += '--cached_vocab {}_vocab.json --word2vec_binary GoogleNews-vectors-negative300.bin '.format(dset)
                    cmd += '--cached_word_embeddings {} --output {} '.format(dset, output_f)
                    cmd += '--checkpoint_dir {}/{} '.format('paper_checkpoints', identifier_str)
                    
                    if os.path.exists(output_f): continue
                    all_commands.append(cmd)
                    
files = [open('{}_commands.txt'.format(idx+1), 'w') for idx in range(n_parallel)]
idx = 0
for cmd in all_commands:
    files[idx].write(cmd + '\n')
    idx += 1
    idx = idx % n_parallel

for f in files:
    f.close()
