# -*- coding: utf-8 -*-

"""
Description: Collection of useful functions for ML.
"""
import os
import math
import time
import types
import subprocess

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


def folder2df(fpath=None, extension='jpg'):
    if fpath is not None:
        fnames = sorted(fpath.glob(f'**/*.{extension}'))
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


def balance_dataset(df_orig, target_col=None, alpha=0.5):

    """
    Returns more balanced version of dataset by replicating the observations of
    imbalanced classes.
    """

    df = df_orig.copy(deep=True)
    obs_dict = df[target_col].value_counts().to_dict()
    max_obs = max(obs_dict.values())

    for k, v in obs_dict.items():
        t = math.ceil((max_obs * alpha) / v)
        if t > 1:
            df_tmp = pd.DataFrame(columns=df.columns)
            df_append = df[df[target_col] == k]
            for i in range(t - 1):
                df_tmp = df_tmp.append(df_append, ignore_index=True)
            df_tmp = df_tmp.iloc[np.random.permutation(len(df_tmp))]
            df_tmp.reset_index(drop=True, inplace=True)
            df = df.append(df_tmp[:int((t - 1) * v)], ignore_index=True)
        else:
            pass
    np.random.seed(42)
    df = df.iloc[np.random.permutation(len(df))]
    return df.reset_index(drop=True)


def call_subprocess(bash_cmd):
    """
    ex: args = call_subprocess('libreoffice --convert-to pdf fname.docx_')
    """

    process = subprocess.Popen(bash_cmd.split(' '), stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    process.wait()
    return True


skip_instances_tuple = (types.BuiltinFunctionType, types.MethodType,
                        types.BuiltinMethodType, types.FunctionType)


def print_object_attrs(obj):

    object_attributes = [atr for atr in obj.__dir__() if not atr.startswith('__')]

    for atr in object_attributes:
        t = getattr(obj, atr)
        if not isinstance(t, skip_instances_tuple):
            print(f"{atr}: {getattr(obj, atr)} \n")
