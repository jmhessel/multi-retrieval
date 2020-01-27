'''
Code to accompany
"Unsupervised Discovery of Multimodal Links in Multi-Sentence/Multi-Image Documents."
https://github.com/jmhessel/multi-retrieval

This is a work-in-progress TF2.0 port.
'''
import argparse
import collections
import json
import tensorflow as tf
import numpy as np
import os
import sys
import tqdm
import text_utils
import image_utils
import eval_utils
import model_utils
import training_utils
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
                        default='AP',
                        choices=['DC','TK','AP'],
                        type=str)
    parser.add_argument('--sim_mode_k',
                        help='If --sim_mode=TK/AP, what should the k be? '
                        'k=-1 for dynamic = min(n_images, n_sentences))? '
                        'if k > 0, then k=ceil(1./k * min(n_images, n_sentences))',
                        default=-1,
                        type=float)
    parser.add_argument('--lr',
                        type=float,
                        help='Starting learning rate',
                        default=.00005)
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
    parser.add_argument('--dropout',
                        type=float,
                        default=0.5,
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
                        default=.00000001,
                        help='What learning rate decay factor should we use?')
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
                        help='Should we save the train/val/test predictions? '
                        'If so --- they will be saved in this directory.')
    parser.add_argument('--image_model_checkpoint',
                        type=str,
                        default=None,
                        help='If set, the image model will be initialized from '
                        'this model checkpoint.')
    parser.add_argument('--text_model_checkpoint',
                        type=str,
                        default=None,
                        help='If set, the text model will be initialized from '
                        'this model checkpoint.')

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


def main():
    args = parse_args()
    np.random.seed(args.seed)

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
        we_init = text_utils.get_word2vec_matrix(
            word2idx, args.cached_word_embeddings, args.word2vec_binary)
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

    # (n docs, n sent, max n words,)
    text_inp = tf.keras.layers.Input((None, args.seq_len))

    # this input tells you how many sentences are really in each doc
    text_n_inp = tf.keras.layers.Input((1,), dtype='int32')
    if args.end2end:
        # (n docs, n image, x, y, color)
        img_inp = tf.keras.layers.Input((None, 224, 224, 3))
    else:
        # (n docs, n image, feature dim)
        img_inp = tf.keras.layers.Input((None, image_features.shape[1]))
    # this input tells you how many images are really in each doc
    img_n_inp = tf.keras.layers.Input((1,), dtype='int32')

    # Step 2: Define transformations to shared multimodal space.

    # Step 2.1: The text model:
    if args.text_model_checkpoint:
        print('Loading pretrained text model from {}'.format(
            args.text_model_checkpoint))
        single_text_doc_model = tf.keras.models.load_model(args.text_model_checkpoint)
        extracted_text_features = single_text_doc_model(text_inp)
    else:
        word_embedding = tf.keras.layers.Embedding(
            len(word2idx),
            word_emb_dim,
            weights=[we_init] if we_init is not None else None,
            mask_zero=True)
        element_dropout = tf.keras.layers.SpatialDropout1D(args.dropout)
        if args.rnn_type == 'GRU':
            word_rnn = tf.keras.layers.GRU(
                args.joint_emb_dim,
                kernel_initializater=tf.keras.initializers.VarianceScaling(
                    mode='fan_avg',
                    distribution='uniform'))
        else:
            word_rnn = tf.keras.layers.LSTM(args.joint_emb_dim)
        embedded_text_inp = word_embedding(text_inp)
        embedded_text_inp = tf.keras.layers.TimeDistributed(element_dropout)(embedded_text_inp)
        extracted_text_features = tf.keras.layers.TimeDistributed(word_rnn)(embedded_text_inp)
        # extracted_text_features is now (n docs, max n setnences, multimodal dim)
        l2_norm_layer = tf.keras.layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=-1))
        extracted_text_features = l2_norm_layer(extracted_text_features)
        single_text_doc_model = tf.keras.models.Model(
            inputs=text_inp,
            outputs=extracted_text_features)
        
    # Step 2.2: The image model:
    if args.image_model_checkpoint:
        print('Loading pretrained image model from {}'.format(
            args.image_model_checkpoint))
        single_img_doc_model = tf.keras.models.load_model(args.image_model_checkpoint)
        extracted_img_features = single_img_doc_model(img_inp)
    else:
        img_projection = tf.keras.layers.Dense(args.joint_emb_dim)
        if args.end2end:
            from tf.keras.applications.nasnet import NASNetMobile
            cnn = tf.keras.applications.nasnet.NASNetMobile(
                include_top=False, input_shape=(224, 224, 3), pooling='avg')

            extracted_img_features = tf.keras.layers.TimeDistributed(cnn)(img_inp)
            if args.dropout > 0.0:
                extracted_img_features = tf.keras.layers.TimeDistributed(
                    tf.keras.layers.Dropout(args.dropout))(extracted_img_features)
            extracted_img_features = keras.layers.TimeDistributed(img_projection)(
                extracted_img_features)
        else:
            extracted_img_features = tf.keras.layers.Masking()(img_inp)
            if args.dropout > 0.0:
                extracted_img_features = tf.keras.layers.TimeDistributed(
                    tf.keras.layers.Dropout(args.dropout))(extracted_img_features)
            extracted_img_features = tf.keras.layers.TimeDistributed(
                img_projection)(extracted_img_features)

        # extracted_img_features is now (n docs, max n images, multimodal dim)
        l2_norm_layer = tf.keras.layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=-1))
        extracted_img_features = l2_norm_layer(extracted_img_features)
        single_img_doc_model = tf.keras.models.Model(
            inputs=img_inp,
            outputs=extracted_img_features)

    # Step 3: Extract/stack the non-padding image/sentence representations
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
    text_enc = mask_slice_and_stack([extracted_text_features, text_n_inp])
    img_enc = mask_slice_and_stack([extracted_img_features, img_n_inp])

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
        sol = tf.numpy_function(bipartite_match_fn, [sim_matrix, k], tf.int32)
        return tf.reduce_mean(tf.gather_nd(sim_matrix, sol))

    if args.sim_mode == 'DC':
        sim_fn = DC_sim
    elif args.sim_mode == 'TK':
        sim_fn = TK_sim
    elif args.sim_mode == 'AP':
        sim_fn = AP_sim
    else:
        raise NotImplementedError('{} is not implemented sim function'.format(args.sim_fn))

    def make_sims(inp):
        # CHECK ME --- tf.transpose
        sims = tf.keras.backend.dot(inp[0], tf.keras.backend.transpose(inp[1]))
        return sims

    all_sims = make_sims([text_enc, img_enc])
    get_pos_neg_sims = model_utils.make_get_pos_neg_sims(
        args,
        sim_fn)
    
    pos_sims, neg_img_sims, neg_text_sims = tf.keras.layers.Lambda(
        get_pos_neg_sims)([all_sims, text_n_inp, img_n_inp])

    def margin_output(inp):
        pos_s, neg_s = inp
        return tf.math.maximum(neg_s - pos_s + args.margin, 0)

    neg_img_hinge = margin_output([pos_sims, neg_img_sims])
    neg_text_hinge = margin_output([pos_sims, neg_text_sims])

    if args.neg_mining == 'negative_sample':
        pool_fn = lambda x: tf.reduce_mean(x, axis=1, keepdims=True)
    elif args.neg_mining == 'hard_negative':
        pool_fn = lambda x: tf.reduce_max(x, axis=1, keepdims=True)
    else:
        raise NotImplementedError('{} is not a valid for args.neg_mining'.format(args.neg_mining))

    neg_img_loss = tf.keras.layers.Lambda(pool_fn, name='neg_img')(neg_img_hinge)
    neg_text_loss = tf.keras.layers.Lambda(pool_fn, name='neg_text')(neg_text_hinge)

    inputs = [text_inp,
              img_inp,
              text_n_inp,
              img_n_inp]

    model = tf.keras.models.Model(inputs=inputs,
                                  outputs=[neg_img_loss, neg_text_loss])

    opt = tf.keras.optimizers.Adam(args.lr)

    def identity(y_true, y_pred):
        del y_true
        # CHECK
        return tf.reduce_mean(y_pred, axis=-1)

    model.compile(opt, loss=identity)

    if args.test_eval > 0:
        train = train[:args.test_eval * args.docs_per_batch]
        val = val[:args.test_eval * args.docs_per_batch]
        
    train_seq = training_utils.DocumentSequence(
        train,
        image_features,
        image_idx2row,
        max_n_sentence,
        max_n_image,
        word2idx,
        args=args,
        shuffle_docs=True,
        shuffle_sentences=False,
        shuffle_images=True)

    val_seq = training_utils.DocumentSequence(
        val,
        image_features,
        image_idx2row,
        max_n_sentence,
        max_n_image,
        word2idx,
        args=args,
        augment=False,
        shuffle_sentences=False,
        shuffle_docs=False,
        shuffle_images=False)

    sdm = training_utils.SaveDocModels(
        args.checkpoint_dir,
        single_text_doc_model,
        single_img_doc_model)
    
    callbacks = [tf.keras.callbacks.ReduceLROnPlateau(factor=args.lr_decay,
                                                      patience=args.lr_patience,
                                                      min_lr=args.min_lr,
                                                      verbose=True), sdm]


    if args.print_metrics:
        metrics_printer = training_utils.PrintMetrics(
            val,
            image_features,
            image_idx2row,
            word2idx,
            single_text_doc_model,
            single_img_doc_model,
            args)
        callbacks.append(metrics_printer)
        
    history = model.fit(
        train_seq,
        epochs=args.n_epochs,
        validation_data=val_seq,
        callbacks=callbacks)

    if args.output:
        best_image_model_str, best_sentence_model_str, best_logs, best_epoch = sdm.best_checkpoints_and_logs
        single_text_doc_model = tf.keras.models.load_model(best_sentence_model_str)
        single_image_doc_model = tf.keras.models.load_model(best_image_model_str)

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
        if args.print_metrics:
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
