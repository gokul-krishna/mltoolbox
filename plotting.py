from .basic import *

import itertools
import seaborn as sns
from PIL import Image
from matplotlib.colors import LogNorm

# set styling options similar to R's ggplot
plt.style.use('ggplot')


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


def save_spectrogram(data, fpath, log_col=False, log_freq=False):
    """
    Objective  : creates and saves the spectrogram
    Input      : 2D power value of spectrogram
    """
    plt.ioff()
    fig, ax = plt.subplots(1)
    fig.set_size_inches(5, 5)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.axis('off')
    if log_freq:
        plt.semilogy()
    if log_col:
        sns.heatmap(data, norm=LogNorm(data.min(), data.max()), cbar=False)
    else:
        sns.heatmap(data, cbar=False)
    fig.savefig(fpath, dpi=100, frameon='false')
    plt.close(fig)
    return True


def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
