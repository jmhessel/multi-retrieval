'''
for i in predictions/train/*; do python visualize_predictions.py $i\/doc.json $i/pred_weights.npy $PWD/data/wiki/ prediction_dir/$i ; done;
'''
import argparse
import numpy as np
import bipartite_utils
import json
import os
import subprocess


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('document')
    parser.add_argument('predictions')
    parser.add_argument('image_dir')
    parser.add_argument('output')
    return parser.parse_args()


def call(x):
    subprocess.call(x, shell=True)


def main():
    args = parse_args()
    pred_adj = np.load(args.predictions)
    with open(args.document) as f:
        data = json.loads(f.read())

    cur_images = [d[0] for d in data[0]]
    cur_text = [d[0] for d in data[1]]

    
    solve_fn = bipartite_utils.generate_fast_hungarian_solving_function()
    sol = solve_fn(pred_adj, max(*pred_adj.shape))
    scores = pred_adj[sol[:,0], sol[:,1]]
    lines = []
    images_to_copy = []
    for img_idx, sent_idx, sc in sorted(
            zip(sol[:,1], sol[:,0], scores), key=lambda x:-x[-1]):
        lines.append('<p>{} ({:.1f})</p>'.format(cur_text[sent_idx],
                                                 sc*100))
        images_to_copy.append('{}/{}'.format(args.image_dir, cur_images[img_idx]))
        lines.append('<p><img src="{}"></p><p></p>'.format(
            cur_images[img_idx].split('/')[-1]))
        print(cur_text[sent_idx])
        print(cur_images[img_idx])
        print(sc)

    lines.append('<p>Article Text:</p>')
    for sent in cur_text:
        lines.append('<p>{}</p>'.format(sent))

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    with open('{}/view.html'.format(args.output), 'w') as f:
        f.write('\n'.join(lines))

    for im in images_to_copy:
        call('cp {} {}'.format(im, args.output))
    print()

    
    
if __name__ == '__main__':
    main()
