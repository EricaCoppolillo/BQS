import pickle
import numpy as np
import settings
import json
import torch

def load_dataset(fname, to_pickle=True):
    if to_pickle:
        with open(fname + '.pickle', 'rb') as fp:
            return pickle.load(fp)
    else:
        with open(fname + '.json') as fp:
            return json.load(fp)


def compute_max_y_aux_popularity():
    domain = np.linspace(0, 1, 1000)
    codomain = [y_aux_popularity(x) for x in domain]
    max_y_aux_popularity = max(codomain)
    return max_y_aux_popularity


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def y_aux_popularity(x):
    f = 1 / (settings.metrics_beta * np.sqrt(2 * np.pi))
    y = np.tanh(settings.metrics_alpha * x) + \
        settings.metrics_scale * f * np.exp(
        -1 / (2 * (settings.metrics_beta ** 2)) * (x - settings.metrics_percentile) ** 2)
    return y


def y_popularity(x):
    max_y_aux_popularity = compute_max_y_aux_popularity()
    y = y_aux_popularity(x) / max_y_aux_popularity
    return y


def y_position(x, cutoff):
    y = sigmoid(-x * settings.metrics_gamma / cutoff) + 0.5
    return y


def y_custom(popularity, position, cutoff):
    y = y_popularity(popularity) * y_position(position, cutoff)
    return y

def naive_sparse2tensor(data):
    return torch.FloatTensor(data.toarray() if hasattr(data, 'toarray') else data)