# -*- coding: utf-8 -*-

"""
Description: Collection of useful functions for ML.
"""
import os
import math
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fs.osfs import OSFS
from fs.memoryfs import MemoryFS
from fs.copy import copy_fs

from IPython.display import display
from fastprogress import progress_bar, master_bar
from concurrent.futures import ProcessPoolExecutor, as_completed

# set styling options similar to R's ggplot
plt.style.use('ggplot')
# setting image size for jupyter notebook env
plt.rcParams['figure.figsize'] = [15.0, 6.0]


def display_all(df):
    with pd.option_context("display.max_rows", 1000,
                           "display.max_columns", 1000):
        display(df)


# Efficiency
def parallel(func, job_list, n_jobs=16):
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        futures = [pool.submit(func, job) for job in job_list]
        for f in progress_bar(as_completed(futures), total=len(job_list)):
            pass
    return [f.result() for f in futures]


# In-Memory file system
def to_ram(dir_path):
    mem_fs = MemoryFS()
    with OSFS(dir_path) as data_fs:
        copy_fs(data_fs, mem_fs, workers=8)
    return mem_fs


def get_ram_file(fname, mem_fs):
    return mem_fs.openbin(fname)


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
    train.reset_index(inplace=True, drop=True)
    valid.reset_index(inplace=True, drop=True)
    return train, valid


# decorator method for time profiling
def print_time(method):
    def timed(*args, **kw):
        tik = time.time()
        result = method(*args, **kw)
        tok = time.time()
        print(f"{method.__name__} {((tok - tik)):.2f} s")
        return result
    return timed
