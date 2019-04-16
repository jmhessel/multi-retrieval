import numpy as np
from lapjv import lapjv

def generate_fast_hungarian_solving_function():
    def base_solve(W):
        orig_shape = W.shape
        if orig_shape[0] != orig_shape[1]:
            if orig_shape[0] > orig_shape[1]:
                pad_idxs = [[0, 0], [0, W.shape[0]-W.shape[1]]]
                col_pad = True
            else:
                pad_idxs = [[0, W.shape[1]-W.shape[0]], [0, 0]]
                col_pad = False
            W = np.pad(W, pad_idxs, 'constant', constant_values=-100)
        sol, _, cost = lapjv(-W)

        i_s = np.arange(len(sol))
        j_s = sol[i_s]

        sort_idxs = np.argsort(-W[i_s, j_s])
        i_s, j_s = map(lambda x: x[sort_idxs], [i_s, j_s])

        if orig_shape[0] != orig_shape[1]:
            if col_pad:
                valid_idxs = np.where(j_s < orig_shape[1])[0]
            else:
                valid_idxs = np.where(i_s < orig_shape[0])[0]
            i_s, j_s = i_s[valid_idxs], j_s[valid_idxs]

        indices = np.hstack([np.expand_dims(i_s, -1), np.expand_dims(j_s, -1)]).astype(np.int32)
        return indices

    def hungarian_solve(W, k, max_val=1000):
        min_dim = min(*W.shape)
        if k <= 0 or k >= min_dim:
            return base_solve(W)

        add_rows = W.shape[0] > W.shape[1]
        add_len = min_dim-k

        if add_rows:
            add_mat = np.zeros((add_len, W.shape[1]))
            add_mat[:] = max_val
            new_W = np.vstack([W, add_mat])
        else:
            add_mat = np.zeros((W.shape[0], add_len))
            add_mat[:] = max_val
            new_W = np.hstack([W, add_mat])

        indices = base_solve(new_W)
        indices = indices[-k:, :]
        return indices

    return hungarian_solve
