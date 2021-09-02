import numpy as np
import skimage.transform


def image_preprocessing(img):
    img = skimage.transform.resize(img, resolution)
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    return img