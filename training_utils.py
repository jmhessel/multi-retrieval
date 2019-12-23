import tensorflow as tf
import numpy as np
import eval_utils
import text_utils
import image_utils


class DocumentSequence(tf.keras.utils.Sequence):
    def __init__(self,
                 data_in,
                 image_matrix,
                 image_idx2row,
                 max_sentences_per_doc,
                 max_images_per_doc,
                 vocab,
                 args=None,
                 augment=True,
                 shuffle_sentences=False,
                 shuffle_images=True,
                 shuffle_docs=True,
                 force_exact_batch=False):
        self.data_in = data_in
        self.image_matrix = image_matrix
        self.image_idx2row = image_idx2row
        self.max_sentences_per_doc = max_sentences_per_doc
        self.max_images_per_doc = max_images_per_doc
        self.vocab = vocab
        self.args = args
        self.argument = augment
        self.shuffle_sentences = shuffle_sentences
        self.shuffle_images = shuffle_images
        self.shuffle_docs = shuffle_docs
        self.force_exact_batch = force_exact_batch

        
    def __len__(self):
        return int(np.ceil(len(self.data_in) / self.args.docs_per_batch))

    
    def __getitem__(self, idx):
        start = idx * self.args.docs_per_batch
        end = (idx + 1) * self.args.docs_per_batch
        cur_doc_b = self.data_in[start: end]

        images, texts = [], []
        image_n_docs, text_n_docs = [], []
        for idx, vers in enumerate(cur_doc_b):
            cur_images = [img[0] for img in vers[0]]
            cur_text = [text[0] for text in vers[1]]

            if self.shuffle_sentences and not (self.args and self.args.subsample_text > 0):
                np.random.shuffle(cur_text)

            if self.shuffle_images and not (self.args and self.args.subsample_image > 0):
                np.random.shuffle(cur_images)

            if self.args and self.args.subsample_image > 0:
                np.random.shuffle(cur_images)
                cur_images = cur_images[:self.args.subsample_image]

            if self.args and self.args.subsample_text > 0:
                np.random.shuffle(cur_text)
                cur_text = cur_text[:self.args.subsample_text]

            if self.args.end2end:
                cur_images = image_utils.images_to_images(cur_images, augment, args)
                if self.args and self.args.subsample_image > 0:
                    image_padding = np.zeros(
                        (self.args.subsample_image - cur_images.shape[0], 224, 224, 3))
                else:
                    image_padding = np.zeros(
                        (self.max_images_per_doc - cur_images.shape[0], 224, 224, 3))
            else:
                cur_images = image_utils.images_to_matrix(
                    cur_images, self.image_matrix, self.image_idx2row)
                if self.args and self.args.subsample_image > 0:
                    image_padding = np.zeros(
                        (self.args.subsample_image - cur_images.shape[0], cur_images.shape[-1]))
                else:
                    image_padding = np.zeros(
                        (self.max_images_per_doc - cur_images.shape[0], cur_images.shape[-1]))

            cur_text = text_utils.text_to_matrix(cur_text, self.vocab, max_len=self.args.seq_len)

            image_n_docs.append(cur_images.shape[0])
            text_n_docs.append(cur_text.shape[0])

            if self.args and self.args.subsample_text > 0:
                text_padding = np.zeros(
                    (self.args.subsample_text - cur_text.shape[0], cur_text.shape[-1]))
            else:
                text_padding = np.zeros(
                    (self.max_sentences_per_doc - cur_text.shape[0], cur_text.shape[-1]))

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

        if not self.force_exact_batch or len(texts) == self.args.docs_per_batch:
            return ([texts,
                     images,
                     text_n_docs,
                     image_n_docs], y)
        
    def on_epoch_end(self):
        if self.shuffle_docs:
            np.random.shuffle(data_in)


class SaveDocModels(tf.keras.callbacks.Callback):

    def __init__(self,
                 checkpoint_dir,
                 single_text_doc_model,
                 single_image_doc_model):
        super(SaveDocModels, self).__init__()
        self.checkpoint_dir = checkpoint_dir
        self.single_text_doc_model = single_text_doc_model
        self.single_image_doc_model = single_image_doc_model
        
        
    def on_train_begin(self, logs={}):
        self.best_val_loss = np.inf
        self.best_checkpoints_and_logs = None

    def on_epoch_end(self, epoch, logs):
        if logs['val_loss'] < self.best_val_loss:
            print('New best val loss: {:.5f}'.format(logs['val_loss']))
            self.best_val_loss = logs['val_loss']
        else:
            return
        image_model_str = self.checkpoint_dir + '/image_model_epoch_{}_val={:.5f}.model'.format(epoch, logs['val_loss'])
        sentence_model_str = self.checkpoint_dir + '/text_model_epoch_{}_val={:.5f}.model'.format(epoch, logs['val_loss'])
        self.best_checkpoints_and_logs = (image_model_str, sentence_model_str, logs, epoch)

        self.single_text_doc_model.save(sentence_model_str, overwrite=True, save_format='h5')
        self.single_image_doc_model.save(image_model_str, overwrite=True, save_format='h5')


class PrintMetrics(tf.keras.callbacks.Callback):
    def __init__(self,
                 val,
                 image_features,
                 image_idx2row,
                 word2idx,
                 single_text_doc_model,
                 single_img_doc_model,
                 args):
        super(PrintMetrics, self).__init__()
        self.val = val
        self.image_features = image_features
        self.image_idx2row = image_idx2row
        self.word2idx = word2idx
        self.single_text_doc_model = single_text_doc_model
        self.single_img_doc_model = single_img_doc_model
        self.args = args
        
    def on_train_begin(self, logs=None):
        self.epoch = []
        self.history = {}

    def on_epoch_end(self, epoch, logs):
        metrics = eval_utils.print_all_metrics(
            self.val,
            self.image_features,
            self.image_idx2row,
            self.word2idx,
            self.single_text_doc_model,
            self.single_img_doc_model,
            self.args)
        self.epoch.append(epoch)
        for k, v in metrics.items():
            self.history.setdefault(k, []).append(v)
