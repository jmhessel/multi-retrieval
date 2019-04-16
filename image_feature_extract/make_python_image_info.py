import json
import sys
import numpy as np
import tqdm


if len(sys.argv) != 3:
    print('usage: [text file of features] [filenames text file]')
    quit()


def load_matrix(fname, n_rows):
    vecs = []
    with open(fname) as f:
        for idx in tqdm.tqdm(range(n_rows)):
            vecs.append(np.array([float(x) for x in f.readline().split()]))
    return np.vstack(vecs)


def load_ordered_ids(fname):
    ids = []
    with open(fname) as f:
        for line in f:
            if line.strip():
                ids.append(line.split('/')[-1].split('.')[0])
    return ids


ordered_ids = load_ordered_ids(sys.argv[2])     
id2row = {str(idx): row for row, idx in enumerate(ordered_ids)}
print('loading {} rows of matrix...'.format(len(ordered_ids)))
m_mat = load_matrix(sys.argv[1], len(ordered_ids))

with open('id2row.json', 'w') as f:
    f.write(json.dumps(id2row))

np.save(sys.argv[1].split('.')[0], m_mat)

