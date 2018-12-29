from .basic import np
import cv2
import math
import random
from PIL import ExifTags

import jpeg4py as jpeg


def get_exif(im, remove_binary=True):
    """returns dict of exif (meta) data if present"""
    if hasattr(im, '_getexif'):
        exif = {ExifTags.TAGS[k]: v
                for k, v in im._getexif().items()
                if k in ExifTags.TAGS
                }
        if remove_binary:
            exif = {k: exif[k] for k in exif if type(exif[k]) is not bytes}
        return exif
    else:
        return {}


def imread_fast(fname):
    """
    same as imread or Image.open but 6x faster
    installations: > sudo apt-get install libturbojpeg
                   > pip install jpeg4py
    input : file path as string
    output: numpy array (HxWx3)
    """
    return jpeg.JPEG(fname).decode()


def imread(fname):
    im = cv2.imread(str(fname))
    # openCV by default uses BGR ordering but we need RBG usually
    # height x width x channels
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def crop(im, r, c, height, width):
    """crop a image given left top pixel location and target size"""
    return im[r:r + height, c:c + width]


def center_crop(im, min_sz=None):
    """ Returns a center crop of an image"""
    r, c, _ = im.shape
    if min_sz is None:
        min_sz = min(r, c)
    start_r = math.ceil((r - min_sz) / 2)
    start_c = math.ceil((c - min_sz) / 2)
    return crop(im, start_r, start_c, min_sz, min_sz)


def random_crop(im, height, width):
    """ Returns a random crop of a image"""
    r, c, _ = im.shape
    start_r = math.floor(random.uniform(0, r - height))
    start_c = math.floor(random.uniform(0, c - width))
    return crop(im, start_r, start_c, height, width)


def random_hflip(im):
    if np.random.rand() >= .5:
        return np.fliplr(im).copy()
    return im


def random_vflip(im):
    if np.random.rand() >= .5:
        return np.flipud(im).copy()
    return im


stats_dict = {
    'image_net': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
}


def normalize_image(im, stat_type='image_net'):
    stats = stats_dict[stat_type]
    return (im - stats[0]) / stats[1]
