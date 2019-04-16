import numpy as np
from keras.preprocessing import image
from keras.applications.nasnet import preprocess_input
import scipy.misc

data_augmentor = image.ImageDataGenerator(rotation_range=20,
                                          horizontal_flip=True,
                                          fill_mode='reflect')

def crop(image_in, targ, center=False):
    x, y = image_in.shape[:2]
    if center:
        x_off, y_off = (x-targ) // 2, (y-targ) // 2
    else:
        x_off, y_off = np.random.randint(0, (x-targ)), np.random.randint(0, (y-targ))
    res = image_in[x_off:x_off+targ, y_off:y_off+targ, ...]
    return res
    

def augment_image(image_in):
    image_out = data_augmentor.random_transform(image_in)
    return np.expand_dims(crop(image_out, 224), axis=0)


def images_to_matrix(image_list, image_matrix, image_idx2row):
    rows = []
    for img in image_list:
        rows.append(image_idx2row[str(img)])

    if type(image_matrix) is list:
        image_matrix_choices = np.random.choice(len(image_matrix), size=len(rows))
        all_features = [image_matrix[image_matrix_choices[idx]][r,:] for idx, r in enumerate(rows)]
        return np.vstack(all_features)
    else:
        return image_matrix[np.array(rows), :]


def load_images(image_list, preprocess=True, target_size=(224, 224)):
    images = []
    for i in image_list:
        c_img = np.expand_dims(image.img_to_array(
            image.load_img(i, target_size=target_size)), axis=0)
        images.append(c_img)
    images = np.vstack(images)
    if preprocess:
        images = preprocess_input(images)
    return images

    
def images_to_images(image_list, augment, args):
    if args.full_image_paths:
        images = load_images([args.image_dir + '/' + img for img in image_list], target_size=(256, 256))
    else:
        images = load_images([args.image_dir + '/' + img + '.jpg' for img in image_list], target_size=(256, 256))
    if augment:
        images = np.vstack(list(map(augment_image, images)))
    else:
        images = np.vstack(list(map(lambda x: np.expand_dims(crop(x, 224, center=True), axis=0), images)))
    return images
