'''
Model utility functions
'''
import tensorflow as tf
import collections
import sys


def make_get_pos_neg_sims(args, sim_fn):

    def get_pos_neg_sims(inp):
        '''
        Applies the similarity function between all text_idx, img_idx pairs.

        inp is a list of three arguments:
           - sims: the stacked similarity matrix
           - text_n_inp: how many sentences are in each document
           - img_n_inp: how many images are in each document
        '''

        sims, text_n_inp, img_n_inp = inp
        text_index_borders = tf.dtypes.cast(tf.cumsum(text_n_inp), tf.int32)
        img_index_borders = tf.dtypes.cast(tf.cumsum(img_n_inp), tf.int32)
        zero = tf.expand_dims(tf.expand_dims(tf.constant(0, dtype=tf.int32), axis=-1), axis=-1)

        # these give the indices of the borders between documents in our big sim matrix...
        text_index_borders = tf.concat([zero, text_index_borders], axis=0)
        img_index_borders = tf.concat([zero, img_index_borders], axis=0)

        doc2pos_sim = {}
        doc2neg_img_sims = collections.defaultdict(list)
        doc2neg_text_sims = collections.defaultdict(list)

        # for each pair of text set and image set...
        for text_idx in range(args.docs_per_batch):
            for img_idx in range(args.docs_per_batch):
                text_start = tf.squeeze(text_index_borders[text_idx])
                text_end = tf.squeeze(text_index_borders[text_idx+1])
                img_start = tf.squeeze(img_index_borders[img_idx])
                img_end = tf.squeeze(img_index_borders[img_idx+1])
                cur_sims = sims[text_start:text_end, img_start:img_end]
                sim = sim_fn(cur_sims)
                if text_idx == img_idx: # positive cases
                    doc2pos_sim[text_idx] = sim
                else: # negative cases
                    doc2neg_img_sims[text_idx].append(sim)
                    doc2neg_text_sims[img_idx].append(sim)

        pos_sims, neg_img_sims, neg_text_sims = [], [], []
        for idx in range(args.docs_per_batch):
            pos_sims.append(doc2pos_sim[idx])
            neg_img_sims.append(tf.stack(doc2neg_img_sims[idx]))
            neg_text_sims.append(tf.stack(doc2neg_text_sims[idx]))

        pos_sims = tf.expand_dims(tf.stack(pos_sims), -1)
        neg_img_sims = tf.stack(neg_img_sims)
        neg_text_sims = tf.stack(neg_text_sims)
        return [pos_sims, neg_img_sims, neg_text_sims]

    return get_pos_neg_sims
