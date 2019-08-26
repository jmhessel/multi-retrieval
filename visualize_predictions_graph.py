'''
for i in predictions/test/*; do python visualize_predictions.py $i\/doc.json $i/pred_weights.npy prediction_dir/$i ; done;
'''
import argparse
import numpy as np
import bipartite_utils
import json
import os
import subprocess
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('document')
    parser.add_argument('predictions')
    parser.add_argument('output')
    parser.add_argument('--n_to_show', default=5)
    return parser.parse_args()


def call(x):
    subprocess.call(x, shell=True)


def main():
    args = parse_args()
    pred_adj = np.load(args.predictions)
    with open(args.document) as f:
        data = json.loads(f.read())

    images, text = data[0], data[1]

    solve_fn = bipartite_utils.generate_fast_hungarian_solving_function()
    sol = solve_fn(pred_adj, args.n_to_show)
    scores = pred_adj[sol[:,0], sol[:,1]]

    true_adj = np.zeros((len(text), len(images)))
    for text_idx, t in enumerate(text):
        if t[1] == -1: continue
        true_adj[text_idx, t[1]] = 1
    for image_idx, t in enumerate(images):
        if t[1] == -1: continue
        true_adj[t[1], image_idx] = 1

    auc = 100 * roc_auc_score(true_adj.flatten(),
                              pred_adj.flatten())
    print('AUC: {:.2f} {}'.format(auc,
                                  data[-1]))
    
    ordered_images, ordered_sentences = [], []
    for img_idx, sent_idx, sc in sorted(
            zip(sol[:,1], sol[:,0], scores), key=lambda x:-x[-1])[:args.n_to_show]:
        ordered_images.append(img_idx)
        ordered_sentences.append(sent_idx)
        print(sc)

    pred_adj_subgraph = pred_adj[np.array(ordered_sentences),:][:,np.array(ordered_images)]
    true_adj_subgraph = true_adj[np.array(ordered_sentences),:][:,np.array(ordered_images)]
    selected_images = [images[img_idx][0] for img_idx in ordered_images]
    selected_sentences = [text[sent_idx][0] for sent_idx in ordered_sentences]

    # normalize predicted sims to have max 1 and min 0
    # first, clip out negative values
    pred_adj_subgraph = np.clip(pred_adj_subgraph, 0, 1.0)
    pred_adj_subgraph -= np.min(pred_adj_subgraph.flatten())
    pred_adj_subgraph /= np.max(pred_adj_subgraph.flatten())
    assert np.min(pred_adj_subgraph.flatten()) == 0.0
    assert np.max(pred_adj_subgraph.flatten()) == 1.0

    print(pred_adj_subgraph.shape)
    print(ordered_images)
    print(ordered_sentences)
    print(selected_images)
    print(selected_sentences)

    # each line has ((x1, y1, x2, y2), strength, correctness)
    # images go above text
    lines_to_plot = []
    image_text_gap = 2
    same_mode_gap = 2
    offdiag_alpha_mul = .5
    
    def cosine_to_width(cos, exp=2.0, maxwidth=8.0):
        return cos**exp * maxwidth
    def cosine_to_alpha(cos, exp=1/2., maxalpha=1.0):
        return cos**exp * maxalpha

    correct_color, incorrect_color = '#1b7837', '#762a83'
    lines_to_plot = []
    for text_idx in range(args.n_to_show):
        for image_idx in range(args.n_to_show):
            coords = (text_idx*same_mode_gap, 0, image_idx*same_mode_gap, image_text_gap)
            strength = max(pred_adj_subgraph[text_idx, image_idx], 0)
            correctness = true_adj_subgraph[text_idx, image_idx] == 1
            lines_to_plot.append((coords, strength, correctness))

    plt.figure(figsize=(args.n_to_show*same_mode_gap, image_text_gap))
    for (x1, y1, x2, y2), strength, correct in sorted(lines_to_plot,
                                                      key=lambda x: x[1]):
        if x1 == x2: continue
        plt.plot([x1, x2], [y1, y2],
                 linewidth=cosine_to_width(strength),
                 alpha=cosine_to_alpha(strength) * offdiag_alpha_mul,
                 color=correct_color if correct else incorrect_color)
    for (x1, y1, x2, y2), strength, correct in sorted(lines_to_plot,
                                                      key=lambda x: x[1]):
        if x1 != x2: continue
        plt.plot([x1, x2], [y1, y2],
                 linewidth=cosine_to_width(strength),                 
                 color=correct_color if correct else incorrect_color)
    plt.axis('off')
    plt.tight_layout()
    
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    with open(args.output + '/sentences.txt', 'w') as f:
        f.write('\n'.join([' '.join(s.split()) for s in selected_sentences]))
    with open(args.output + '/images.txt', 'w') as f:
        f.write('\n'.join(selected_images))
    with open(args.output + '/all_sentences.txt', 'w') as f:
        f.write('\n'.join([' '.join(s[0].split()) for s in text]))
    with open(args.output + '/all_images.txt', 'w') as f:
        f.write('\n'.join([x[0] for x in images]))
    with open(args.output + '/auc.txt', 'w') as f:
        f.write('{:.4f}'.format(auc))
    plt.savefig(args.output + '/graph.png', dpi=300)
    call('convert {} -trim {}'.format(args.output + '/graph.png',
                                      args.output + '/graph_cropped.png'))
    
    
if __name__ == '__main__':
    main()
