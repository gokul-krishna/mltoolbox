# modified functions from fast.ai
import torch
import torch.nn.functional as F


def n_class_accuracy(y_pred, y_true):
    """Compute accuracy with `y_true` when `y_pred`
       is of batch_size * n_classes shape"""
    n = y_true.shape[0]
    y_pred = y_pred.argmax(dim=-1).view(n, -1)
    y_true = y_true.view(n, -1)
    return (y_pred == y_true).float().mean()


def binary_accuracy(y_pred, y_true, thresh=0.5, sigmoid=False):
    "Compute accuracy when `y_pred` and `y_true` are the same size."
    if sigmoid:
        y_pred = y_pred.sigmoid()
    return ((y_pred > thresh) == y_true.byte()).float().mean()


def top_k_accuracy(y_pred, y_true, k=5):
    "Computes the Top-k accuracy (y_true is in the top k predictions)."
    n = y_true.shape[0]
    y_pred = y_pred.topk(k=k, dim=-1)[1].view(n, -1)
    y_true = y_true.view(n, -1)
    return (y_pred == y_true).sum(dim=1, dtype=torch.float32).mean()


def error_rate(y_pred, y_true):
    "1 - `accuracy`"
    return 1 - n_class_accuracy(y_pred, y_true)


def mean_absolute_error(y_pred, y_true):
    "Mean absolute error between `y_pred` and `y_true`."
    return torch.abs(y_true - y_pred).mean()


def mean_squared_error(y_pred, y_true):
    "Mean squared error between `y_pred` and `y_true`."
    return F.mse_loss(y_pred, y_true)


def root_mean_squared_error(y_pred, y_true):
    "Root mean squared error between `y_pred` and `y_true`."
    return torch.sqrt(F.mse_loss(y_pred, y_true))


def mean_squared_logarithmic_error(y_pred, y_true):
    "Mean squared logarithmic error between `y_pred` and `y_true`."
    return F.mse_loss(torch.log(1 + y_pred), torch.log(1 + y_true))
