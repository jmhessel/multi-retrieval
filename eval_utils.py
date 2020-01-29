import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
import sacrebleu

# pycocoevalcap imports
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider


import collections
import tqdm
import image_utils
import text_utils
import os
import sklearn.preprocessing
import json


def compute_match_metrics_doc(docs,
                              image_matrix,
                              image_idx2row,
                              vocab,
                              text_trans,
                              image_trans,
                              args,
                              ks=[1,2,3,4,5,6,7,8,9,10]):

    all_aucs, all_match_metrics = [], collections.defaultdict(list)
    all_texts, all_images = [], []


    #for MT metrics
    all_refs = []
    all_sys = []
    
    # supress warwning that I believe is thrown by keras' timedistributed
    # over dynamic tensors...
    tf.get_logger().setLevel('ERROR')
    for images, text, meta in tqdm.tqdm(docs):
        # text by image matrix
        cur_text = [t[0] for t in text]
        cur_images = [t[0] for t in images]
        n_text, n_image = map(len, [cur_text, cur_images])

        cur_text = text_utils.text_to_matrix(cur_text, vocab, max_len=args.seq_len)

        if args.end2end:
            cur_images = image_utils.images_to_images(cur_images, False, args)
        else:
            cur_images = image_utils.images_to_matrix(cur_images, image_matrix, image_idx2row)

        cur_text = np.expand_dims(cur_text, 0)
        cur_images = np.expand_dims(cur_images, 0)
    
        text_vec = text_trans.predict(cur_text)
        image_vec = image_trans.predict(cur_images)

        text_vec = text_vec[0,:n_text,:]
        image_vec = image_vec[0,:n_image,:]
        
        pred_adj = text_vec.dot(image_vec.transpose())
        true_adj = np.zeros((len(text), len(images)))

        # for MT metrics, for each image with a ground-truth sentence,
        # extract predicted sentences.
        im2best_text_idxs = np.argmax(pred_adj, axis=0)
        im2all_predicted_captions = [text[idx][0] for idx in im2best_text_idxs]
        
        im2predicted_captions = {}
        im2ground_truth_captions = collections.defaultdict(list)
        
        for text_idx, t in enumerate(text):
            if t[1] == -1: continue
            true_adj[text_idx, t[1]] = 1
        for image_idx, t in enumerate(images):
            if t[1] == -1: continue
            im2predicted_captions[image_idx] = im2all_predicted_captions[image_idx]
            im2ground_truth_captions[image_idx].append(text[t[1]][0])
            true_adj[t[1], image_idx] = 1

        for img_idx, pred in im2predicted_captions.items():
            all_refs.append(im2ground_truth_captions[img_idx])
            all_sys.append(pred)
       
        pred_adj = pred_adj.flatten()
        true_adj = true_adj.flatten()

        if np.sum(true_adj) == 0:
            print('Skipping no ground truth edges...')
            continue
        if np.sum(true_adj) == len(true_adj):
            print('Skipping only truth edges...')
            continue

        pred_order = true_adj[np.argsort(-pred_adj)]
        for k in ks:
            all_match_metrics[k].append(np.mean(pred_order[:k]))

        all_aucs.append(roc_auc_score(true_adj, pred_adj))
    tf.get_logger().setLevel('INFO')

    n_refs_max = np.max([len(r) for r in all_refs])
    
    all_refs_final = []
    for outer_idx in range(n_refs_max):
        cur_refs = [all_refs[inner_idx][min(outer_idx, len(all_refs[inner_idx])-1)]
                    for inner_idx in range(len(all_refs))]
        all_refs_final.append(cur_refs)
    
    all_mt_metrics = compute_mt_metrics(all_sys, all_refs_final)
    return all_aucs, all_match_metrics, all_mt_metrics


def compute_mt_metrics(all_sys, all_refs):
    res_dict = {}
    sacre_bleu = sacrebleu.corpus_bleu(all_sys, all_refs)
    res_dict['sacre_bleu'] = sacre_bleu.score
    
    try:
        tokenizer = PTBTokenizer()

        def tokenize_list(lst_in):
            tmp_dict = {idx: [{'caption':v}] for idx, v in enumerate(lst_in)}
            tmp_dict = tokenizer.tokenize(tmp_dict)
            return [tmp_dict[idx][0] for idx in range(len(lst_in))]

        res = tokenize_list(all_sys)
        gts = [tokenize_list(r) for r in all_refs]

        # we need a dictionary mapping
        # unique_idx --> pred and unique_idx --> [gts]
        res_dict_coco = {idx: [r] for idx, r in enumerate(res)}
        gts_dict_coco = collections.defaultdict(list)
        for lst in gts:
            for idx, cap in enumerate(lst):
                gts_dict_coco[idx].append(cap)

        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr")
        ]

        for scorer, method in scorers:
            score, _ = scorer.compute_score(gts_dict_coco, res_dict_coco)
            if type(method) is list:
                for s, m in zip(score, method):
                    res_dict['MSCOCO_{}'.format(m)] = s
            else:
                res_dict['MSCOCO_{}'.format(method)] = score
    except Exception as e:
        print('Unable to compute MSCOCO metrics: {}'.format(e))
        print('continuing nonetheless')

    return res_dict


def compute_feat_spread(feats_in):
    feats_in = sklearn.preprocessing.normalize(feats_in)
    mean = np.expand_dims(np.mean(feats_in, axis=0), 0)
    feats_in = feats_in - mean
    norms = np.linalg.norm(feats_in, axis=1)
    norms = norms * norms
    return np.mean(norms)


def save_predictions(docs,
                     image_matrix,
                     image_idx2row,
                     vocab,
                     text_trans,
                     image_trans,
                     output_dir,
                     args,
                     limit=None):

    if limit is not None:
        docs = docs[:limit]
    
    for idx, (images, text, meta) in tqdm.tqdm(enumerate(docs)):
        # text by image matrix
        identifier = meta if meta else str(idx)
        identifier = meta.replace('/', '_')
        
        cur_text = [t[0] for t in text]
        cur_images = [t[0] for t in images]
        
        n_text, n_image = map(len, [cur_text, cur_images])

        cur_text = text_utils.text_to_matrix(cur_text, vocab, max_len=args.seq_len)

        if args.end2end:
            cur_images = image_utils.images_to_images(cur_images, False, args)
        else:
            cur_images = image_utils.images_to_matrix(cur_images, image_matrix, image_idx2row)

        cur_images = np.expand_dims(cur_images, 0)
        cur_text = np.expand_dims(cur_text, 0)

        text_vec = text_trans.predict_on_batch(cur_text) 
        image_vec = image_trans.predict_on_batch(cur_images)

        text_vec = text_vec[0,:n_text,:]
        image_vec = image_vec[0,:n_image,:]

        image_spread, text_spread = map(compute_feat_spread,
                                        [text_vec, image_vec])

        cur_out_dir = output_dir + '/{}_textspread_{:.4f}_imagespread_{:.4f}/'.format(
            identifier, text_spread, image_spread)

        if not os.path.exists(cur_out_dir):
            os.makedirs(cur_out_dir)
        pred_adj = text_vec.dot(image_vec.transpose())
        np.save(cur_out_dir + '/pred_weights.npy',
                pred_adj)
        with open(cur_out_dir + '/doc.json', 'w') as f:
            f.write(json.dumps((images, text, meta)))


def print_all_metrics(data,
                      image_features,
                      image_idx2row,
                      word2idx,
                      single_text_doc_model,
                      single_img_doc_model,
                      args):
    metrics_dict = {}
    aucs, rec2prec, mt_metrics = compute_match_metrics_doc(data,
                                                           image_features,
                                                           image_idx2row,
                                                           word2idx,
                                                           single_text_doc_model,
                                                           single_img_doc_model,
                                                           args)
    print('Validation AUC={:.2f}'.format(100*np.mean(aucs)))
    metrics_dict['aucs'] = 100 * np.mean(aucs)
    prec = {}
    prec_str = 'Validation '
    ks = sorted(list(rec2prec.keys()))
    for k in ks:
        res = np.mean(rec2prec[k])*100
        metrics_dict['p@{}'.format(k)] = res
        prec_str += 'p@{}={:.2f} '.format(k, res)
    print(prec_str.strip())
    print('Machine translation metrics: {}'.format(str(mt_metrics)))
    metrics_dict['mt_metrics'] = mt_metrics
    return metrics_dict
