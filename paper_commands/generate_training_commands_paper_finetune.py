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

n_parallel = 1

all_commands = []

for batch_size, dset, alg, subsample_img, subsample_txt, neg_mining in [
        (11, 'rqa', 'AP', 10, 10, 'hard_negative'),
        (11, 'diy', 'AP', 10, 10, 'hard_negative'),
        (11, 'wiki', 'AP', 10, 10, 'hard_negative')]: 
    identifier_str = '{}+{}+{}+{}+{}+FT'.format(dset, batch_size, alg, -1, neg_mining)
    output_f = 'paper_res/{}.pkl'.format(identifier_str)
    #if os.path.exists(output_f): continue
    if 'wiki' != dset:
        cmd = 'python3 train_doc.py data/{}/docs.json --image_dir data/{}/images '.format(dset, dset)
    else:
        cmd = 'python3 train_doc.py data/{}/docs.json --image_dir data/{} '.format(dset, dset)
    cmd += '--sim_mode {} --neg_mining {} --sim_mode_k {} '.format(alg, neg_mining, -1)
    cmd += '--cached_vocab {}_vocab.json --word2vec_binary GoogleNews-vectors-negative300.bin '.format(dset)
    cmd += '--cached_word_embeddings {} --output {} '.format(dset, output_f)
    cmd += '--checkpoint_dir {}/{} '.format('paper_checkpoints', identifier_str)
    cmd += '--subsample_text {} '.format(subsample_txt)
    cmd += '--subsample_image {} '.format(subsample_img)
    cmd += '--end2end 1 '
    cmd += '--seq_len 50 '
    cmd += '--force 1 '
    if dset == 'wiki':
        cmd += '--full_image_paths 1'
    all_commands.append(cmd)
        
files = [open('{}_commands_FT.txt'.format(idx+1), 'w') for idx in range(n_parallel)]
idx = 0
for cmd in all_commands:
    files[idx].write(cmd + '\n')
    idx += 1
    idx = idx % n_parallel

for f in files:
    f.close()
