# -*- coding: utf-8 -*-

"""
@author: Gokul

Description : My collection of useful functions for Machine Learning.
"""

import cv2
import math
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from fastprogress import progress_bar
from concurrent.futures import ProcessPoolExecutor, as_completed


# set styling options similar to R's ggplot
plt.style.use('ggplot')


# Plotting
def multi_plot(fnames, ncols=3):
    """
        Display multiple images in a grid structure
        fnames: list of file name with full/relative path
    """
    assert ncols != 0
    no_imgs = len(fnames)
    nrows = math.ceil(no_imgs / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 3 * nrows),
                             subplot_kw={'xticks': [], 'yticks': []})
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    for ax, fname in zip(axes.flat, fnames):
        im = Image.open(fname)
        ax.imshow(im)
        ax.set_title(fname.split('/')[-1], size=20)
    plt.tight_layout()
    plt.show()


# Images
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


def random_crop(x, height, width):
    """ Returns a random crop of a image"""
    r, c, _ = x.shape
    start_r = math.floor(random.uniform(0, r - height))
    start_c = math.floor(random.uniform(0, c - width))
    return crop(x, start_r, start_c, height, width)


stats_dict = {
    'image_net': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
}


def noralize_image(im, stat_type='image_net'):
    stats = stats_dict[stat_type]
    return (im - stats[0]) / stats[1]


# Efficiency
def parallel(func, job_list, n_jobs=16):
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        futures = [pool.submit(func, job) for job in job_list]
        for f in progress_bar(as_completed(futures), total=len(job_list)):
            pass
    return [f.result() for f in futures]


def folder2df(fpath=None):
    if fpath is not None:
        fnames = sorted(fpath.glob('**/*.jpg'))
        df = pd.DataFrame(fnames, columns=['fname'])
        df['label'] = df['fname'].apply(lambda x: str(x).split('/')[-2]
                                        ).astype('category')
        return df


def split_df(df, train_ratio=0.8):
    np.random.seed(42)
    mask = np.random.random(df.shape[0]) < train_ratio
    train = df[mask].copy()
    valid = df[~mask].copy()
    return train, valid
