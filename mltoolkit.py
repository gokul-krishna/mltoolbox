# -*- coding: utf-8 -*-

"""
@author: Gokul

Description : My collection of useful functions for Machine Learning.
"""

import math
from PIL import Image
import matplotlib.pyplot as plt


# set styling options similar to R's ggplot
plt.style.use('ggplot')


def multi_plot(fnames, ncols=3):
    """
        Display multiple images in a grid structure

        fnames: list of file name with full path
    """
    no_imgs = len(fnames)
    ncols = 3
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
