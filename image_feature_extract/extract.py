from keras.applications.densenet import preprocess_input as preprocess_input_dn
from keras.applications.densenet import DenseNet169

from keras.preprocessing import image
from keras.models import Sequential, Model
from keras.layers import Flatten
import os, sys, numpy as np
import keras
import tensorflow as tf
import keras.backend as K


def load_images(image_list):
    images = []
    for i in image_list:
        c_img = np.expand_dims(image.img_to_array(
            image.load_img(i, target_size = (256, 256))), axis=0)
        images.append(c_img)
    return np.vstack(images)


def image_generator(fnames, batch_size):
    cfns = []
    for i, p in enumerate(fnames):
        cfns.append(p)
        if len(cfns) == batch_size:
            yield load_images(cfns)
            cfns = []
    if len(cfns) != 0:
        yield load_images(cfns)
        cfns = []

            
if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("usage: [image list] [output name]")
        quit()


    gpu_memory_frac = .45
    if gpu_memory_frac > 0 and gpu_memory_frac < 1:
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_frac
        session = tf.Session(config=config)
        K.set_session(session)
        
    keras.backend.set_learning_phase(0)
    do_center_crop = True

    file_list = sys.argv[1]
    fpath = sys.argv[2]

    all_paths = []
    with open(file_list) as f:
        for i,line in enumerate(f):
            all_paths.append(line.strip())
            
    print("Extracting from {} files".format(len(all_paths)))
            
    print('Saving to {}'.format(fpath))
    base_model = DenseNet169(include_top=False, input_shape=(224,224,3))
    base_model.trainable = False
    base_model.summary()

    m_image = keras.layers.Input((256, 256, 3), dtype='float32')

    if do_center_crop:
        crop = keras.layers.Lambda(lambda x: tf.image.central_crop(x, .875),
                                          output_shape=(224, 224, 3))(m_image)
    else:
        crop = keras.layers.Lambda(lambda x: tf.map_fn(lambda y: tf.random_crop(y, [224, 224, 3]), x),
                                   output_shape=(224, 224, 3))(m_image)

    crop = keras.layers.Lambda(lambda x: preprocess_input_dn(x))(crop)
    trans = base_model(crop)
    trans = keras.layers.GlobalAveragePooling2D()(trans)
    model = Model(inputs=m_image,
                  outputs=trans)
    model.summary()

    # model = keras.utils.multi_gpu_model(model, 2)
    print('moved the image model to the multi gpus!')
    batch_size = 100
    print('Getting started...')
    gen = image_generator(all_paths, batch_size)
    feats = model.predict_generator(gen,
                                    np.ceil(len(all_paths)/batch_size),
                                    use_multiprocessing=True,
                                    workers=1,
                                    verbose=1)
    print(feats.shape)
    print("Saving to {}".format(fpath))
    with open(fpath, 'w') as f:
        for i in range(feats.shape[0]):
            f.write(" ".join(["{:.10f}".format(x) for x in feats[i,:]]) + "\n")
