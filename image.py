from .basic import np, plt
import cv2
import math
import random
import PIL
import torch

import jpeg4py as jpeg


def get_exif(im, remove_binary=True):
    """returns dict of exif (meta) data if present"""
    assert type(im) == PIL.JpegImagePlugin.JpegImageFile
    if hasattr(im, '_getexif') and im._getexif() is not None:
        exif = {PIL.ExifTags.TAGS[k]: v
                for k, v in im._getexif().items()
                if k in PIL.ExifTags.TAGS
                }
        if remove_binary:
            exif = {k: exif[k] for k in exif if type(exif[k]) is not bytes}
        return exif
    else:
        return {}


def imread_fast(fname):
    """
    same as imread or Image.open but 6x faster
    installations: > apt-get install libturbojpeg
                   > brew install libjpeg-turbo / libjpeg
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


def im_int2float(im):
    return im.astype(np.float32) / 255


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


def random_hflip(im, prob=0.5):
    if np.random.rand() <= prob:
        return np.fliplr(im).copy()
    return im


def random_vflip(im, prob=0.5):
    if np.random.rand() <= prob:
        return np.flipud(im).copy()
    return im


stats_dict = {
    'image_net': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
}


def normalize_image(im, stat_type='image_net'):
    stats = stats_dict[stat_type]
    return (im - stats[0]) / stats[1]


def pil2cv(im):
    return np.array(im)


def vcyclic_shift(im, alpha=0.5, no_blocks=1):
    h = im.shape[0]
    if no_blocks > 1:
        block_len = round(h / no_blocks)
        s = np.random.randint(0, int(alpha * no_blocks))
        part = int(s * block_len)
    else:
        s = np.random.uniform(0, alpha)
        part = int(h * s)
    im_ = im[:part, :]
    _im = im[-h + part:, :]
    return np.concatenate([_im, im_], axis=0)


def hcyclic_shift(im, alpha=0.5, no_blocks=1):
    w = im.shape[1]
    if no_blocks > 1:
        block_len = round(w / no_blocks)
        s = np.random.randint(0, int(alpha * no_blocks))
        part = int(s * block_len)
    else:
        s = np.random.uniform(0, alpha)
        part = int(w * s)
    im_ = im[:, :part]
    _im = im[:, -w + part:]
    return np.concatenate([_im, im_], axis=1)


def im2tensor(im):
    return torch.tensor(np.rollaxis(im, 2), dtype=torch.float32)


def create_bb_rect(bb, color='red'):
    """creates a rect bounding box, used in bounding box visualization"""
    ymin, xmin, ymax, xmax = bb
    bb = np.array(bb, dtype=np.float32)
    return plt.Rectangle(xy=(xmin, ymin), width=(xmax - xmin),
                         height=(ymax - ymin), color=color,
                         fill=False, lw=3)


def show_bb(im, bb):
    """show image with bounding box"""
    plt.imshow(im)
    plt.gca().add_patch(create_bb_rect(bb))


def random_crop_bb(im, height, width, bb):
    r, c, _ = im.shape
    ymin, xmin, ymax, xmax = bb
    start_r = math.floor(random.uniform(max(0, (ymax - height)),
                                        min(ymin, (r - height))))
    start_c = math.floor(random.uniform(max(0, (xmax - width)),
                                        min(xmin, (c - width))))
    new_bb = [(ymin - start_r), (xmin - start_c),
              (ymax - start_r), (xmax - start_c)]
    return crop(im, start_r, start_c, height, width), new_bb


class InvalidInputException(Exception):
    pass


def resize(im, new_height=None, new_width=None, scale=0.5):
    """resizes images"""
    r, c, _ = im.shape

    if new_height is None and new_width is None and scale is not None:
        # keeping the same aspect ratio as original
        new_height = int(scale * r)
        new_width = int(scale * c)
    elif new_height is None and new_width is not None:
        # use the scale based on old and new width
        scale = float(new_width) / float(c)
        new_height = int(scale * r)
    elif new_height is not None and new_width is None:
        # use the scale based on old and new height
        scale = float(new_height) / float(r)
        new_width = int(scale * c)
    elif new_height is not None and new_width is not None:
        # just use the new height and old height
        pass
    else:
        raise InvalidInputException('Invalid input configuration')

    imr = cv2.resize(im, (new_width, new_height))
    return imr


def resize_bb(im, bb, new_height=None, new_width=None, scale=0.5):
    """resizes image and bounding box together"""
    r, c, _ = im.shape
    ymin, xmin, ymax, xmax = bb
    if new_height is None and new_width is None and scale is not None:
        # keeping the same aspect ratio as original
        new_height = int(scale * r)
        new_width = int(scale * c)
    elif new_height is None and new_width is not None:
        # use the scale based on old and new width
        scale = float(new_width) / float(c)
        new_height = int(scale * r)
    elif new_height is not None and new_width is None:
        # use the scale based on old and new height
        scale = float(new_height) / float(r)
        new_width = int(scale * c)
    elif new_height is not None and new_width is not None:
        # just use the new height and old height
        pass
    else:
        raise InvalidInputException('Invalid input configuration')

    imr = cv2.resize(im, (new_width, new_height))
    new_bb = [int((ymin / r) * new_height), int((xmin / c) * new_width),
              int((ymax / r) * new_height), int((xmax / c) * new_width)]
    return imr, new_bb


def imsave(im, fname, extension='.jpg'):
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    return cv2.imwrite(fname + extension, im)
