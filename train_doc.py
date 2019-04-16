'''
Code to accompany
"Unsupervised Discovery of Multimodal Links in Multi-Sentence/Multi-Image Documents."
https://github.com/jmhessel/multi-retrieval
'''
import argparse
import collections
import json
import keras
import keras.backend as K
import tensorflow as tf
import numpy as np
import os
import sys
import tqdm
import text_utils
import image_utils
import eval_utils
import bipartite_utils
import pickle
from pprint import pprint


def load_data(fname):
    with open(fname) as f:
        return json.loads(f.read())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('documents',
                        help='json of train/val/test documents.')
    parser.add_argument('--image_features',
                        help='path to pre-extracted image-feature numpy array.')
    parser.add_argument('--image_id2row',
                        help='path to mapping from image id --> numpy row for image features.')
    parser.add_argument('--joint_emb_dim',
                        type=int,
                        help='Embedding dimension of the shared, multimodal space.',
                        default=1024)
    parser.add_argument('--margin',
                        type=float,
                        help='Margin for computing hinge loss.',
                        default=.2)
    parser.add_argument('--seq_len',
                        type=int,
                        help='Maximum token sequence length for each sentence before truncation.',
                        default=20)
    parser.add_argument('--docs_per_batch',
                        type=int,
                        help='How many docs per batch? 11 docs = 10 negative samples per doc.',
                        default=11)
    parser.add_argument('--neg_mining',
                        help='What type of negative mining?',
                        default='hard_negative',
                        choices=['negative_sample', 'hard_negative'],
                        type=str)
    parser.add_argument('--sim_mode',
                        help='What similarity function should we use?',
                        default='DC',
                        choices=['DC','TK','AP'],
                        type=str)
    parser.add_argument('--sim_mode_k',
                        help='If --sim_mode=TK/AP, what should the k be? k=-1 for dynamic = min(n_images, n_sentences))? '
                        'if k > 0, then k=ceil(1./k * min(n_images, n_sentences))',
                        default=-1,
                        type=float)
    parser.add_argument('--lr',
                        type=float,
                        help='Starting learning rate',
                        default=.0001)
    parser.add_argument('--n_epochs',
                        type=int,
                        help='How many epochs to run for?',
                        default=50)
    parser.add_argument('--checkpoint_dir',
                        type=str,
                        help='What directory to save checkpoints in?',
                        default='checkpoints')
    parser.add_argument('--word2vec_binary',
                        type=str,
                        help='If cached word embeddings have not been generated, '
                        'what is the location of the word2vec binary?',
                        default=None)
    parser.add_argument('--cached_word_embeddings',
                        type=str,
                        help='Where are/will the cached word embeddings saved?',
                        default='cached_word2vec.json')
    parser.add_argument('--print_metrics',
                        type=int,
                        help='Should we print the metrics if there are ground-truth '
                        'labels, or no?',
                        default=0)
    parser.add_argument('--cached_vocab',
                        type=str,
                        help='Where should we cache the vocab, if anywhere '
                        '(None means no caching)',
                        default=None)
    parser.add_argument('--output',
                        type=str,
                        default=None,
                        help='If output is set, we will save a pkl file'
                        'with the validation/test metrics.')
    parser.add_argument('--seed',
                        type=int,
                        help='Random seed',
                        default=1)
    parser.add_argument('--gpu_memory_frac',
                        type=float,
                        default=1.0,
                        help='How much of the GPU should we allow tensorflow to allocate?')
    parser.add_argument('--dropout',
                        type=float,
                        default=0.4,
                        help='How much dropout should we apply?')
    parser.add_argument('--subsample_image',
                        type=int,
                        default=-1,
                        help='Should we subsample images to constant lengths? '
                        'This option is useful if the model is being trained end2end '
                        'and there are memory issues.')
    parser.add_argument('--subsample_text',
                        type=int,
                        default=-1,
                        help='Should we subsample sentences to constant lengths? '
                        'This option is useful if the model is being trained end2end '
                        'and there are memory issues.')
    parser.add_argument('--rnn_type',
                        type=str,
                        default='GRU',
                        help='What RNN should we use')
    parser.add_argument('--end2end',
                        type=int,
                        default=0,
                        help='Should we backprop through the whole vision network?')
    parser.add_argument('--image_dir',
                        type=str,
                        default=None,
                        help='What image dir should we use, if end2end?')
    parser.add_argument('--lr_patience',
                        type=int,
                        default=3,
                        help='What learning rate patience should we use?')
    parser.add_argument('--lr_decay',
                        type=float,
                        default=.2,
                        help='What learning rate decay factor should we use?')
    parser.add_argument('--min_lr',
                        type=float,
                        default=.0000001,
                        help='What learning rate decay factor should we use?')
    parser.add_argument('--val_loss_mean_neg_sample',
                        type=int,
                        default=0,
                        help='Instead of tracking hard negative validation loss, '
                        'should we track top-5 mean negative sample loss?')
    parser.add_argument('--full_image_paths',
                        type=int,
                        default=0,
                        help='For end2end training, should we use full image paths '
                        '(i.e., is the file extention already on images?)?')
    parser.add_argument('--test_eval',
                        type=int,
                        help='(DEBUG OPTION) If test_eval >= 1, then training '
                        'only happens over this many batches',
                        default=-1)
    parser.add_argument('--force',
                        type=int,
                        default=0,
                        help='Should we force the run if the output exists?')
    parser.add_argument('--save_predictions',
                        type=str,
                        default=None,
                        help='Should we save the train/val/test predictions? If so --- they will be saved in this directory.')
    parser.add_argument('--image_model_checkpoint',
                        type=str,
                        default=None,
                        help='If set, the image model will be initialized from this model checkpoint.')
    parser.add_argument('--text_model_checkpoint',
                        type=str,
                        default=None,
                        help='If set, the text model will be initialized from this model checkpoint.')

    args = parser.parse_args()

    # check to make sure that various flags are set correctly
    if args.end2end:
        assert args.image_dir is not None
    if not args.end2end:
        assert args.image_features is not None and args.image_id2row is not None

    # print out some info about the run's inputs/outputs
    if args.output and '.pkl' not in args.output:
        args.output += '.pkl'

    if args.output:
        print('Output will be saved to {}'.format(args.output))
    print('Model checkpoints will be saved in {}'.format(args.checkpoint_dir))

    if args.output and os.path.exists(args.output) and not args.force:
        print('{} already done! If you want to force it, set --force 1'.format(args.output))
        quit()

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    if args.save_predictions:
        if not os.path.exists(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)
            os.makedirs(args.checkpoint_dir + '/train')
            os.makedirs(args.checkpoint_dir + '/val')
            os.makedirs(args.checkpoint_dir + '/test')

    return args


def training_data_generator(data_in,
                            image_matrix,
                            image_idx2row,
                            max_sentences_per_doc,
                            max_images_per_doc,
                            vocab,
                            seq_len,
                            augment=True,
                            docs_per_batch=30,
                            args=None,
                            shuffle_sentences=False,
                            shuffle_images=True,
                            shuffle_docs=True,
                            run_forever=True,
                            force_exact_batch=False):
    iter_num = 0
    while True:
        cur_start_idx = 0
        cur_end_idx = docs_per_batch
        while cur_start_idx < cur_end_idx:
            cur_doc_b = data_in[cur_start_idx:cur_end_idx]
            images, texts = [], []
            image_n_docs, text_n_docs = [], []
            for idx, vers in enumerate(cur_doc_b):
                cur_images = [img[0] for img in vers[0]]
                cur_text = [text[0] for text in vers[1]]

                if shuffle_sentences and not (args and args.subsample_text > 0):#in that case, we'll shuffle later...
                    np.random.shuffle(cur_text)

                if shuffle_images and not (args and args.subsample_image > 0):#in that case, we'll shuffle later...
                    np.random.shuffle(cur_images)

                if args and args.subsample_image > 0:
                    # if we are subsampling, we better shuffle...
                    np.random.shuffle(cur_images)
                    cur_images = cur_images[:args.subsample_image]

                if args and args.subsample_text > 0:
                    np.random.shuffle(cur_text)
                    cur_text = cur_text[:args.subsample_text]

                if args.end2end:
                    cur_images = image_utils.images_to_images(cur_images, augment, args)
                    if args and args.subsample_image > 0:
                        image_padding = np.zeros((args.subsample_image - cur_images.shape[0], 224, 224, 3))
                    else:
                        image_padding = np.zeros((max_images_per_doc - cur_images.shape[0], 224, 224, 3))
                else:
                    cur_images = image_utils.images_to_matrix(cur_images, image_matrix, image_idx2row)
                    if args and args.subsample_image > 0:
                        image_padding = np.zeros((args.subsample_image - cur_images.shape[0], cur_images.shape[-1]))
                    else:
                        image_padding = np.zeros((max_images_per_doc - cur_images.shape[0], cur_images.shape[-1]))

                cur_text = text_utils.text_to_matrix(cur_text, vocab, max_len=seq_len)

                image_n_docs.append(cur_images.shape[0])
                text_n_docs.append(cur_text.shape[0])

                if args and args.subsample_text > 0:
                    text_padding = np.zeros((args.subsample_text - cur_text.shape[0], cur_text.shape[-1]))
                else:
                    text_padding = np.zeros((max_sentences_per_doc - cur_text.shape[0], cur_text.shape[-1]))

                cur_images = np.vstack([cur_images, image_padding])
                cur_text = np.vstack([cur_text, text_padding])

                cur_images = np.expand_dims(cur_images, 0)
                cur_text = np.expand_dims(cur_text, 0)

                images.append(cur_images)
                texts.append(cur_text)

            images = np.vstack(images)
            texts = np.vstack(texts)

            image_n_docs = np.expand_dims(np.array(image_n_docs), -1)
            text_n_docs = np.expand_dims(np.array(text_n_docs), -1)

            y = [np.zeros(len(text_n_docs)), np.zeros(len(image_n_docs))]

            if not force_exact_batch or len(texts) == docs_per_batch:
                yield ([texts,
                        images,
                        text_n_docs,
                        image_n_docs], y)
            cur_start_idx = cur_end_idx
            cur_end_idx += docs_per_batch
            cur_end_idx = min(len(data_in), cur_end_idx)

        if shuffle_docs:
            np.random.shuffle(data_in)
        iter_num += 1
        if not run_forever: break


def main():
    args = parse_args()
    np.random.seed(args.seed)

    if args.gpu_memory_frac > 0 and args.gpu_memory_frac < 1:
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = args.gpu_memory_frac
        session = tf.Session(config=config)
        K.set_session(session)

    data = load_data(args.documents)
    train, val, test = data['train'], data['val'], data['test']
    np.random.shuffle(train); np.random.shuffle(val); np.random.shuffle(test)
    max_n_sentence, max_n_image = -1, -1
    for d in train + val + test:
        imgs, sents, meta = d
        max_n_sentence = max(max_n_sentence, len(sents))
        max_n_image = max(max_n_image, len(imgs))

    print('Max n sentence={}, max n image={}'.format(max_n_sentence, max_n_image))
    if args.cached_vocab:
        print('Saving/loading vocab from {}'.format(args.cached_vocab))

    # create vocab from training documents:
    flattened_train_sents = []
    for _, sents, _ in train:
        flattened_train_sents.extend([s[0] for s in sents])
    word2idx = text_utils.get_vocab(flattened_train_sents, cached=args.cached_vocab)
    print('Vocab size was {}'.format(len(word2idx)))

    if args.word2vec_binary:
        we_init = text_utils.get_word2vec_matrix(word2idx, args.cached_word_embeddings, args.word2vec_binary)
    else:
        we_init = np.random.uniform(low=-.02, high=.02, size=(len(word2idx), 300))

    if args.end2end:
        image_features = None
        image_idx2row = None
    else:
        image_features = np.load(args.image_features)
        image_idx2row = load_data(args.image_id2row)

    word_emb_dim = 300

    if val[0][0][0][1] is not None:
        ground_truth = True
        print('The input has ground truth, so AUC will be computed.')
    else:
        ground_truth = False


    # Step 1: Specify model inputs/outputs:

    # (n docs, n sent (dynamic), max n words,)
    text_inp = keras.layers.Input((None, args.seq_len))
    # this input tells you how many sentences are really in each doc
    text_n_inp = keras.layers.Input((1,), dtype='int32')
    if args.end2end:
        # (n docs, n sent (dynamic), x, y, color)
        img_inp = keras.layers.Input((None, 224, 224, 3))
    else:
        # (n docs, n image (dynamic), feature dim)
        img_inp = keras.layers.Input((None, image_features.shape[1]))
    # this input tells you how many images are really in each doc
    img_n_inp = keras.layers.Input((1,), dtype='int32')

    # Step 2: Define transformations to shared multimodal space.

    # Step 2.1: The text model:
    word_embedding = keras.layers.Embedding(len(word2idx),
                                            word_emb_dim,
                                            weights=[we_init] if we_init is not None else None,
                                            mask_zero=True)
    if args.rnn_type == 'GRU':
        word_rnn = keras.layers.GRU(args.joint_emb_dim, recurrent_dropout=args.dropout)
    else:
        word_rnn = keras.layers.LSTM(args.joint_emb_dim, recurrent_dropout=args.dropout)
    embedded_text_inp = word_embedding(text_inp)
    extracted_text_features = keras.layers.TimeDistributed(word_rnn)(embedded_text_inp)
    # extracted_text_features is now (n docs, max n setnences, multimodal dim)

    # Step 2.2: The image model:
    img_projection = keras.layers.Dense(args.joint_emb_dim)
    if args.end2end:
        from keras.applications.nasnet import NASNetMobile
        cnn = keras.applications.nasnet.NASNetMobile(
            include_top=False, input_shape=(224, 224, 3), pooling='avg')

        extracted_img_features = keras.layers.TimeDistributed(cnn)(img_inp)
        if args.dropout > 0.0:
            extracted_img_features = keras.layers.TimeDistributed(
                keras.layers.Dropout(args.dropout))(extracted_img_features)
        extracted_img_features = keras.layers.TimeDistributed(img_projection)(
            extracted_img_features)
    else:
        img_projection = keras.layers.Dense(args.joint_emb_dim)
        extracted_img_features = keras.layers.Masking()(img_inp)
        if args.dropout > 0.0:
            extracted_img_features = keras.layers.TimeDistributed(
                keras.layers.Dropout(args.dropout))(extracted_img_features)
        extracted_img_features = keras.layers.TimeDistributed(
            img_projection)(extracted_img_features)

    # extracted_img_features is now (n docs, max n images, multimodal dim)

    # Step 3: L2 normalize each extracted feature vectors to finish, and
    # define the models that can be run at test-time.
    extracted_text_features = keras.layers.Lambda(
        lambda x: K.l2_normalize(x, axis=-1))(extracted_text_features)
    extracted_img_features = keras.layers.Lambda(
        lambda x: K.l2_normalize(x, axis=-1))(extracted_img_features)
    single_text_doc_model = keras.models.Model(
        inputs=text_inp,
        outputs=extracted_text_features)
    single_img_doc_model = keras.models.Model(
        inputs=img_inp,
        outputs=extracted_img_features)

    if args.text_model_checkpoint:
        print('Loading pretrained text model from {}'.format(
            args.text_model_checkpoint))
        single_text_doc_model = keras.models.load_model(args.text_model_checkpoint)

    if args.image_model_checkpoint:
        print('Loading pretrained image model from {}'.format(
            args.image_model_checkpoint))
        single_img_doc_model = keras.models.load_model(args.image_model_checkpoint)


    # Step 4: Extract/stack the non-padding image/sentence representations
    def mask_slice_and_stack(inp):
        stacker = []
        features, n_inputs = inp
        n_inputs = tf.dtypes.cast(n_inputs, tf.int32)
        # for each document, we will extract the portion of input features that are not padding
        # this means, for features[doc_idx], we will take the first n_inputs[doc_idx] rows.
        # we stack them into one big array so we can do a big cosine sim dot product between all
        # sentence image pairs in parallel. We'll slice up this array back up later.
        for idx in range(args.docs_per_batch):
            cur_valid_idxs = tf.range(n_inputs[idx,0])
            cur_valid_features = features[idx]
            feats = tf.gather(cur_valid_features, cur_valid_idxs)
            stacker.append(feats)
        return tf.concat(stacker, axis=0)

    # extracted text/img features are (n_docs, max_in_seq, dim)
    # we want to compute cosine sims between all (sent, img) pairs quickly
    # so we will stack them into new tensors ...
    # text_enc has shape (total number of sent in batch, dim)
    # img_enc has shape (total number of image in batch, dim)
    text_enc = keras.layers.Lambda(mask_slice_and_stack)([extracted_text_features, text_n_inp])
    img_enc = keras.layers.Lambda(mask_slice_and_stack)([extracted_img_features, img_n_inp])


    def DC_sim(sim_matrix):
        text2im_S = tf.reduce_mean(tf.reduce_max(sim_matrix, 1))
        im2text_S = tf.reduce_mean(tf.reduce_max(sim_matrix, 0))
        return text2im_S + im2text_S

    def get_k(sim_matrix):
        k = tf.minimum(tf.shape(sim_matrix)[0], tf.shape(sim_matrix)[1])
        if args.sim_mode_k > 0:
            k = tf.dtypes.cast(k, tf.float32)
            k = tf.math.ceil(tf.div(k, args.sim_mode_k))
            k = tf.dtypes.cast(k, tf.int32)
        return k

    def TK_sim(sim_matrix):
        k = get_k(sim_matrix)
        im2text_S, text2im_S = tf.reduce_max(sim_matrix, 0), tf.reduce_max(sim_matrix, 1)
        text2im_S = tf.reduce_mean(tf.math.top_k(text2im_S, k=k)[0], axis=-1)
        im2text_S = tf.reduce_mean(tf.math.top_k(im2text_S, k=k)[0], axis=-1)
        return text2im_S + im2text_S

    bipartite_match_fn = bipartite_utils.generate_fast_hungarian_solving_function()
    def AP_sim(sim_matrix):
        k = get_k(sim_matrix)
        sol = tf.py_func(bipartite_match_fn, [sim_matrix, k], tf.int32)
        return tf.reduce_mean(tf.gather_nd(sim_matrix, sol))

    if args.sim_mode == 'DC':
        sim_fn = DC_sim
    elif args.sim_mode == 'TK':
        sim_fn = TK_sim
    elif args.sim_mode == 'AP':
        sim_fn = AP_sim
    else:
        raise NotImplementedError('{} is not implemented sim function'.format(args.sim_fn))

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

    def make_sims(inp):
        sims = K.dot(inp[0], K.transpose(inp[1]))
        return sims

    all_sims = keras.layers.Lambda(make_sims)([text_enc, img_enc])
    pos_sims, neg_img_sims, neg_text_sims = keras.layers.Lambda(
        get_pos_neg_sims)([all_sims, text_n_inp, img_n_inp])

    def margin_output(inp):
        pos_s, neg_s = inp
        return K.maximum(neg_s - pos_s + args.margin, 0)

    neg_img_hinge = keras.layers.Lambda(margin_output)([pos_sims, neg_img_sims])
    neg_text_hinge = keras.layers.Lambda(margin_output)([pos_sims, neg_text_sims])

    if args.neg_mining == 'negative_sample':
        pool_fn = lambda x: tf.reduce_mean(x, axis=1, keepdims=True)
    elif args.neg_mining == 'hard_negative':
        pool_fn = lambda x: tf.reduce_max(x, axis=1, keepdims=True)
    else:
        raise NotImplementedError('{} is not a valid for args.neg_mining'.format(args.neg_mining))

    if args.val_loss_mean_neg_sample and args.neg_mining != 'negative_sample':
        pool_fn_tmp = pool_fn
        # at train time, do hard negative mining. Val time, do top-5 negative mining.
        pool_fn = lambda x: K.in_train_phase(
            pool_fn_tmp(x),
            tf.reduce_mean(tf.math.top_k(x, k=5)[0], axis=1, keepdims=True))

    neg_img_loss = keras.layers.Lambda(pool_fn, name='neg_img')(neg_img_hinge)
    neg_text_loss = keras.layers.Lambda(pool_fn, name='neg_text')(neg_text_hinge)

    inputs = [text_inp,
              img_inp,
              text_n_inp,
              img_n_inp]

    model = keras.models.Model(inputs=inputs,
                               outputs=[neg_img_loss, neg_text_loss])
    model.summary()

    opt = keras.optimizers.Adam(args.lr)

    def identity(y_true, y_pred):
        del y_true
        return K.mean(y_pred, axis=-1)

    model.compile(opt, loss=identity)

    train_gen = training_data_generator(train,
                                        image_features,
                                        image_idx2row,
                                        max_n_sentence,
                                        max_n_image,
                                        word2idx,
                                        args.seq_len,
                                        args=args,
                                        docs_per_batch=args.docs_per_batch,
                                        shuffle_docs=True,
                                        shuffle_sentences=False,
                                        shuffle_images=True,
                                        force_exact_batch=True)

    # pad val with repeated data so that batch sizes are consistent
    n_to_add = args.docs_per_batch - len(val) % args.docs_per_batch
    if n_to_add != args.docs_per_batch:
        print('Padding iterator validation data with {} data points so batch sizes are consistent...'.format(n_to_add))
        padded_val = val + val[:n_to_add]
        assert not len(padded_val) % args.docs_per_batch
    else:
        padded_val = val

    val_gen = training_data_generator(padded_val,
                                      image_features,
                                      image_idx2row,
                                      max_n_sentence,
                                      max_n_image,
                                      word2idx,
                                      args.seq_len,
                                      args=args,
                                      docs_per_batch=args.docs_per_batch,
                                      augment=False,
                                      shuffle_sentences=False,
                                      shuffle_docs=False,
                                      shuffle_images=False,
                                      force_exact_batch=True)

    class SaveDocModels(keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.best_val_loss = np.inf
            self.best_checkpoints_and_logs = None

        def on_epoch_end(self, epoch, logs):
            if logs['val_loss'] < self.best_val_loss:
                print('New best val loss: {:.5f}'.format(logs['val_loss']))
                self.best_val_loss = logs['val_loss']
            else:
                return
            image_model_str = args.checkpoint_dir + '/image_model_epoch_{}_val={:.5f}.model'.format(epoch, logs['val_loss'])
            sentence_model_str = args.checkpoint_dir + '/text_model_epoch_{}_val={:.5f}.model'.format(epoch, logs['val_loss'])
            self.best_checkpoints_and_logs = (image_model_str, sentence_model_str, logs, epoch)
            single_img_doc_model.save(image_model_str, overwrite=True)
            single_text_doc_model.save(sentence_model_str, overwrite=True)

    sdm = SaveDocModels()
    callbacks = [keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                   factor=args.lr_decay,
                                                   patience=args.lr_patience,
                                                   min_lr=args.min_lr,
                                                   verbose=True), sdm]

    class PrintMetrics(keras.callbacks.Callback):
        def on_train_begin(self, logs=None):
            self.epoch = []
            self.history = {}

        def on_epoch_end(self, epoch, logs):
            metrics = eval_utils.print_all_metrics(val,
                                                   image_features,
                                                   image_idx2row,
                                                   word2idx,
                                                   single_text_doc_model,
                                                   single_img_doc_model,
                                                   args)
            self.epoch.append(epoch)
            for k, v in metrics.items():
                self.history.setdefault(k, []).append(v)


    if args.print_metrics:
        metrics_printer = PrintMetrics()
        callbacks.append(metrics_printer)

    history = model.fit_generator(
        train_gen,
        steps_per_epoch=(len(train) // args.docs_per_batch + (0 if len(train) % args.docs_per_batch == 0 else 1)) if args.test_eval <= 0 else args.test_eval,
        epochs=args.n_epochs,
        validation_data=val_gen,
        validation_steps=len(val) // args.docs_per_batch if args.test_eval <= 0 else args.test_eval,
        callbacks=callbacks)

    if args.output:
        best_image_model_str, best_sentence_model_str, best_logs, best_epoch = sdm.best_checkpoints_and_logs

        single_text_doc_model = keras.models.load_model(best_sentence_model_str)
        single_image_doc_model = keras.models.load_model(best_image_model_str)

        if ground_truth:
            val_aucs, val_match_metrics = eval_utils.compute_match_metrics_doc(
                val,
                image_features,
                image_idx2row,
                word2idx,
                single_text_doc_model,
                single_img_doc_model,
                args)

            test_aucs, test_match_metrics = eval_utils.compute_match_metrics_doc(
                test,
                image_features,
                image_idx2row,
                word2idx,
                single_text_doc_model,
                single_img_doc_model,
                args)
        else:
            val_aucs, test_aucs = None, None
            val_match_metrics, test_match_metrics = None, None

        output = {'logs':best_logs,
                  'best_sentence_model_str':best_sentence_model_str,
                  'best_image_model_str':best_image_model_str,
                  'val_aucs':val_aucs,
                  'val_match_metrics':val_match_metrics,
                  'test_aucs':test_aucs,
                  'test_match_metrics':test_match_metrics,
                  'args':args,
                  'epoch':best_epoch}

        for k, v in history.history.items():
            output['history_{}'.format(k)] = v
        for k, v in metrics_printer.history.items():
            output['metrics_history_{}'.format(k)] = v


        with open(args.output, 'wb') as f:
            pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)
        print('saved output to {}'.format(args.output))

    if args.save_predictions:
        for d, name in zip([train, val, test], ['train', 'val', 'test']):
            out_dir = args.save_predictions + '/' + name
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            eval_utils.save_predictions(
                d,
                image_features,
                image_idx2row,
                word2idx,
                single_text_doc_model,
                single_img_doc_model,
                out_dir,
                args)


if __name__ == '__main__':
    main()
