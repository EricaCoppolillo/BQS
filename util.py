import pickle5 as pickle
import numpy as np
import random
import json
import torch
import torch.nn.functional as F
from torch import nn
import re


def load_dataset(fname, to_pickle=True):
    if to_pickle:
        with open(fname + '.pickle', 'rb') as fp:
            return pickle.load(fp)
    else:
        with open(fname + '.json') as fp:
            return json.load(fp)


def compute_max_y_aux_popularity(settings):
    domain = np.linspace(0, 1, 1000)
    codomain = [y_aux_popularity(x, settings) for x in domain]
    max_y_aux_popularity = max(codomain)
    return max_y_aux_popularity


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def y_aux_popularity(x, settings):
    f = 1 / (settings.metrics_beta * np.sqrt(2 * np.pi))
    y = np.tanh(settings.metrics_alpha * x) + \
        settings.metrics_scale * f * np.exp(
        -1 / (2 * (settings.metrics_beta ** 2)) * (x - settings.metrics_percentile) ** 2)
    return y


def y_popularity(x, settings):
    max_y_aux_popularity = compute_max_y_aux_popularity(settings)
    y = y_aux_popularity(x, settings) / max_y_aux_popularity
    return y


def y_position(x, cutoff, settings):
    y = sigmoid(-x * settings.metrics_gamma / cutoff) + 0.5
    return y


def y_custom(popularity, position, cutoff, settings):
    y = y_popularity(popularity, settings) * y_position(position, cutoff, settings)
    return y


def naive_sparse2tensor(data):
    return torch.FloatTensor(data.toarray() if hasattr(data, 'toarray') else data)


def set_seed(seed):
    if not seed:
        seed = 10
    print("[ Using Seed : ", seed, " ]")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clean_json_string(json_str):
    p = re.compile('(?<!\\\\)\'')
    json_str = p.sub('\"', json_str)
    return json_str


def normalize_distr(matrix, method="norm"):

    if method == "softmax":
        return nn.Softmax(dim=1)(matrix)
        #return softmax(matrix, axis=1)
    elif method == "norm":
        return matrix / matrix.sum(axis=1, keepdims=True)
    else:
        raise Exception(f"method you passed: {method} not yet supported. Choose in [softmax, norm].")




def _kl_div_2d(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    # D_KL(P || Q)
    batch, chans, height, width = p.shape
    unsummed_kl = F.kl_div(
        q.reshape(batch * chans, height * width).log(), p.reshape(batch * chans, height * width), reduction='none'
    )
    kl_values = unsummed_kl.sum(-1).view(batch, chans)
    return kl_values


def js_div_2d(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    # JSD(P || Q)
    m = 0.5 * (p + q)
    return 0.5 * _kl_div_2d(p, m) + 0.5 * _kl_div_2d(q, m)