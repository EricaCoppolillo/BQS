# on GPU epinions limit=1, bs=16
SEED = 8734

import collections
import json
import math
import os
import pickle
import random
import types
from datetime import datetime

import numpy as np
from scipy.sparse import lil_matrix

np.random.seed(SEED)
random.seed(SEED)

import matplotlib.pyplot as plt
# %matplotlib inline

import time

import torch.nn as nn
import torch
from torchsummary import summary

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

USE_CUDA = True

CUDA = USE_CUDA and torch.cuda.is_available()

device = torch.device("cuda" if CUDA else "cpu")

print(torch.__version__)

if CUDA:
    print('run on cuda %s' % os.environ['CUDA_VISIBLE_DEVICES'])
else:
    print('cuda not available')

dataset_name = 'ml-1m'
# dataset_name = 'ml-20m'
# dataset_name = 'netflix_sample'
# dataset_name = 'pinterest'
# dataset_name = 'epinions'

print('Dataset:', dataset_name)
data_dir = os.path.expanduser('./data')

data_dir = os.path.join(data_dir, dataset_name)
dataset_file = os.path.join(data_dir, 'data_rvae')
proc_dir = os.path.join(data_dir, 'pg/')
if not os.path.exists(proc_dir):
    os.makedirs(proc_dir)

result_dir = os.path.join(data_dir, 'results')
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# TODO: include learning params
file_model = os.path.join(result_dir, 'best_model.pth')
file_train = os.path.join(proc_dir, 'train_glob')

to_pickle = True

"""## DataLoader 

The dataset is split into train/test with indexes mask

In train the data is reported with a mask of *x* items selected randomly between positive and negative with a proportion 1:2

In test the data is reported with 3 masks of items with less, middle and top popolarity
"""


def load_dataset(fname, to_pickle=True):
    if to_pickle:
        with open(fname + '.pickle', 'rb') as fp:
            return pickle.load(fp)
    else:
        with open(fname + '.json') as fp:
            return json.load(fp)


class DataLoader:
    def __init__(self, file_tr, pos_neg_ratio=4, negatives_in_test=100, use_popularity=False):

        dataset = load_dataset(file_tr)

        self.n_users = dataset['users']
        self.item_popularity = dataset['popularity']
        self.thresholds = dataset['thresholds']
        self.pos_neg_ratio = pos_neg_ratio
        self.negatives_in_test = negatives_in_test

        # IMPROVEMENT
        self.low_pop = len([i for i in self.item_popularity if i <= self.thresholds[0]])
        self.med_pop = len([i for i in self.item_popularity if self.thresholds[0] < i <= self.thresholds[1]])
        self.high_pop = len([i for i in self.item_popularity if self.thresholds[1] < i])

        '''
        if dataset_name == 'ml-20m':
            limit = self.high_pop + int(self.med_pop / 2)
        else:
            limit = self.high_pop
        '''

        # limit = self.high_pop
        limit = 1

        self.n_items = len(self.item_popularity)
        self.use_popularity = use_popularity
        self.sorted_item_popularity = sorted(self.item_popularity)
        self.max_popularity = self.sorted_item_popularity[-limit]
        self.min_popularity = self.sorted_item_popularity[0]
        self.frequencies = [int(round(self.max_popularity * (self.pos_neg_ratio / min(p, self.max_popularity)))) for p in self.item_popularity]
        self.double_frequencies = [self.max_popularity * (self.pos_neg_ratio / min(p, self.max_popularity)) for p in self.item_popularity]

        self.max_width = -1

        print('DATASET STATS ------------------------------')
        print('users:', self.n_users)
        print('items:', self.n_items)
        print('low_pop:', self.low_pop)
        print('med_pop:', self.med_pop)
        print('high_pop:', self.high_pop)
        print('thresholds:', self.thresholds)
        print('max_popularity:', self.max_popularity)
        print('min_popularity:', self.min_popularity)
        print('max_frequency:', max(self.frequencies))
        print('min_frequency:', min(self.frequencies))
        print('num(max_popularity):', sum(self.item_popularity == self.max_popularity))
        print('num(min_popularity):', sum(self.item_popularity == self.min_popularity))
        print('sorted(self.sorted_item_popularity)[:100]:', sorted(self.sorted_item_popularity[:10]))
        print('sorted(self.sorted_item_popularity)[-100:]:', sorted(self.sorted_item_popularity[-10:]))
        print('sorted(frequencies):', sorted(self.frequencies)[:10])

        print('phase 1: Loading data...')
        print('initialize...')
        self._initialize()
        print('load_data...')
        self._load_data(dataset)

        '''
        print('phase 2: Generating training masks...')

        for tag in ('train', 'validation', 'test'):
            self._generate_mask_tr(tag)

            print('SET {}, shape {}, positives {}'.format(tag.upper(), self.data[tag].shape, self.data[tag].sum()))
        '''

        print('phase 3: generating test masks...')

        for tag in ('validation', 'test'):
            self._generate_mask_te(tag)

        print('Done.')

    def _generate_mask_te(self, tag):
        self.mask_rank[tag] = np.zeros((self.size[tag], self.n_items), dtype=np.int8)

        for i in np.arange(self.size[tag]):
            if i % 1000 == 0:
                print('tag {} {}/{}'.format(tag, i, self.size[tag]))
            pos = self.pos_rank[tag][i]
            self.mask_rank[tag][i, pos] = 1

    def _initialize(self):
        self._count_positive = {'train': None, 'validation': None, 'test': None}
        self._count_positive_user = {'train': None, 'validation': None, 'test': None}

        self.data = dict()
        self.pos = dict()  # history of the user
        self.neg = dict()
        self.mask = dict()
        self.pos_rank = dict()  # items to predict
        self.neg_rank = dict()
        self.mask_rank = dict()
        self.size = dict()

        for tag in ('train', 'validation', 'test'):
            self.pos[tag] = []
            self.neg[tag] = []

        for tag in ('validation', 'test'):
            self.pos_rank[tag] = []
            self.neg_rank[tag] = []

    def _load_data(self, dataset):
        train = []
        validation = []
        test = []

        training_data = dataset['training_data']
        validation_data = dataset['validation_data']
        test_data = dataset['test_data']

        counter = 1
        for user_id in training_data:
            pos, neg = training_data[user_id]
            assert len(neg) >= 100, f'fail train neg for user {user_id}'

            items_np = np.zeros(self.n_items, dtype=np.int8)
            items_np[pos] = 1
            train.append(items_np)

            if counter % 1000 == 0:
                print('load_data (train): user {}/{}'.format(counter, int(math.ceil(self.n_users * 0.7))))

            pos, neg = self._generate_training_pairs(pos)

            self.pos['train'].append(pos)
            self.neg['train'].append(neg)

            counter += 1

        for user_id in validation_data:
            positives_tr, positives_te, negatives_sampled = validation_data[user_id]
            if len(positives_te) == 0:
                continue

            assert len(negatives_sampled) >= 100, f'fail valid neg for user {user_id}'

            items_np = np.zeros(self.n_items, dtype=np.int8)
            items_np[positives_tr] = 1
            items_np[positives_te] = 1
            validation.append(items_np)

            if counter % 1000 == 0:
                print('load_data (validation): user {}/{}'.format(counter, int(math.ceil(self.n_users * 0.3))))

            pos, neg = self._generate_pairs(positives_tr)
            self.pos['validation'].append(pos)
            self.neg['validation'].append(neg)

            self.pos_rank['validation'].append(positives_te)
            self.neg_rank['validation'].append(negatives_sampled)

        for user_id in test_data:
            positives_tr, positives_te, negatives_sampled = test_data[user_id]
            if len(positives_te) == 0:
                continue

            assert len(negatives_sampled) >= 100, f'fail test neg for user {user_id}'

            items_np = np.zeros(self.n_items, dtype=np.int8)
            items_np[positives_tr] = 1
            items_np[positives_te] = 1
            test.append(items_np)

            if counter % 1000 == 0:
                print('load_data (test): user {}/{}'.format(counter, int(math.ceil(self.n_users * 0.3))))

            pos, neg = self._generate_pairs(positives_tr)
            self.pos['test'].append(pos)
            self.neg['test'].append(neg)

            self.pos_rank['test'].append(positives_te)
            self.neg_rank['test'].append(negatives_sampled)

        self.data['train'] = np.array(train, dtype=np.int8)
        self.size['train'] = len(train)
        if len(validation) > 0:
            self.data['validation'] = np.array(validation, dtype=np.int8)
            self.size['validation'] = len(validation)
        if len(test) > 0:
            self.data['test'] = np.array(test, dtype=np.int8)
            self.size['test'] = len(test)

    def _sample_negatives(self, pos, size):
        all_items = set(range(len(self.item_popularity)))
        all_negatives = list(all_items - set(pos))

        return random.choices(all_negatives, k=size)

    def _generate_training_pairs(self, pos):
        # IMPROVEMENT
        if self.use_popularity or len(pos) == 1:

            positives = []
            for item in pos:
                frequency = self.frequencies[item]
                positives[0:0] = [item] * frequency

            negatives = self._sample_negatives(pos, len(positives))
            self.max_width = max(self.max_width, len(negatives))

            return positives, negatives
        else:
            self.max_width = self.n_items * self.pos_neg_ratio
            return self._generate_pairs(pos)

    def _generate_pairs(self, pos):
        positives = pos * self.pos_neg_ratio
        # random.shuffle(positives)
        negatives = self._sample_negatives(pos, len(positives))

        return positives, negatives

    def iter(self, batch_size=256, tag='train'):
        """
        Iter on data

        :param batch_size: size of the batch
        :return: batch_idx, x, pos, neg, mask_pos, mask_neg
        """

        assert (tag in ('train', 'validation', 'test'))

        # idxlist = np.arange(self._N)
        idxlist = np.arange(self.data[tag].shape[0])
        np.random.shuffle(idxlist)

        N = idxlist.shape[0]

        for start_idx in range(0, N, batch_size):
            end_idx = min(start_idx + batch_size, N)

            x = self.data[tag][idxlist[start_idx:end_idx]]

            number_of_rows = end_idx - start_idx

            mask = lil_matrix((number_of_rows, self.max_width))
            pos = lil_matrix((number_of_rows, self.max_width))
            neg = lil_matrix((number_of_rows, self.max_width))

            for row in range(number_of_rows):
                length = len(self.pos[tag][start_idx + row])

                pos[row, :length] = self.pos[tag][start_idx + row]
                neg[row, :length] = self.neg[tag][start_idx + row]
                mask[row, :length] = [1] * length

            # pos = coo_matrix(pos.todense())
            # neg = coo_matrix(neg.todense())
            # mask = coo_matrix(mask.todense())

            if (start_idx == 0):
                # print('>>> pos:', pos[0, :10])
                # print('>>> neg:', neg[0, :10])
                # print('>>> mask:', mask[0, :10])
                pass

            yield x, pos, neg, mask

    def iter_test(self, batch_size=256, tag='test'):
        """
        Iter on data

        mask_loss

        :param batch_size: size of the batch
        :return: batch_idx, x, pos, neg, mask_pos, mask_neg
        """

        # assert (tag in ('validation','test'))

        N = self.size[tag]
        for start_idx in range(0, N, batch_size):
            end_idx = min(start_idx + batch_size, N)

            x = self.data[tag][start_idx:end_idx]

            number_of_rows = end_idx - start_idx

            mask = lil_matrix((number_of_rows, self.max_width))
            pos = lil_matrix((number_of_rows, self.max_width))
            neg = lil_matrix((number_of_rows, self.max_width))

            for row in range(number_of_rows):
                length = len(self.pos[tag][start_idx + row])
                pos[row, :length] = self.pos[tag][start_idx + row]
                neg[row, :length] = self.neg[tag][start_idx + row]
                mask[row, :length] = [1] * length

            pos_te = self.pos_rank[tag][start_idx:end_idx]
            neg_te = self.neg_rank[tag][start_idx:end_idx]

            mask_pos_te = self.mask_rank[tag][start_idx:end_idx]

            # pos = coo_matrix(pos.todense())
            # neg = coo_matrix(neg.todense())
            # mask = coo_matrix(mask.todense())
            if (start_idx == 0):
                # print('??? pos', pos[0, :10])
                # print('??? neg:', neg[0, :10])
                # print('??? mask:', mask[0, :10])

                print('??? pos_te[0]:', pos_te[0])
                print('??? neg_te[0]:', neg_te[0])

                print('??? mask_pos_te[0]:', mask_pos_te[0])
                pass

            yield x, pos, neg, mask, pos_te, neg_te, mask_pos_te

    def get_items_counts_by_cat(self, tag):
        """
        Return the number of positive items in test set
        :return: #less, #middle, #most
        """

        assert (tag in ('validation', 'test'))

        if self._count_positive[tag] is None:
            self._count_positive[tag] = [0, 0, 0]

            for pos in self.pos_rank[tag]:
                for item in pos:
                    if self.item_popularity[item] <= self.thresholds[0]:
                        self._count_positive[tag][0] += 1
                    elif self.thresholds[0] < self.item_popularity[item] <= self.thresholds[1]:
                        self._count_positive[tag][1] += 1
                    else:
                        self._count_positive[tag][2] += 1

        return self._count_positive[tag]

    def get_users_counts_by_cat(self, tag):
        """
        Return the number of user in popularity category in test set
        :return: #less, #middle, #most
        """

        assert (tag in ('validation', 'test'))

        if self._count_positive_user[tag] is None:
            self._count_positive_user[tag] = [0, 0, 0]

            for pos in self.pos_rank[tag]:
                match = [False, False, False]
                for item in pos:
                    if self.item_popularity[item] <= self.thresholds[0]:
                        match[0] = True
                    elif self.thresholds[0] < self.item_popularity[item] <= self.thresholds[1]:
                        match[1] = True
                    else:
                        match[2] = True

                    if match[0] and match[1] and match[2]:
                        break

                for i, v in enumerate(match):
                    if v:
                        self._count_positive_user[tag][i] += 1

        return self._count_positive_user[tag]


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
        settings.metrics_scale * f * np.exp(-1 / (2 * (settings.metrics_beta ** 2)) * (x - settings.metrics_percentile) ** 2)
    return y


def y_popularity(x):
    y = y_aux_popularity(x) / max_y_aux_popularity
    return y


def y_position(x, cutoff):
    y = sigmoid(-x * settings.metrics_gamma / cutoff) + 0.5
    return y


def y_custom(popularity, position, cutoff):
    y = y_popularity(popularity) * y_position(position, cutoff)
    return y


def compute_metric(x, y, pos, neg, popularity, popularity_thresholds, top_k=10):
    """
    Compute metric:
    - hitrate
    - hitrate per popularity

    :param x: user preferences, BS x items
    :param y: user predictions, BS x items
    :param mask: user selected preferences, BS x 100
    :param popularity: dict of item popularity (normalized)
    :param top_k: top k items to select ranked by score
    :return: hitrate, popularity hitrate, total_positives(low,medium,high)
    """
    assert min([len(r) for r in neg]) >= top_k and top_k > 0, f"fail with top_k = {top_k} and neg = {[len(r) for r in neg]}"
    assert len(popularity) == y.shape[-1], f'{len(popularity)} != {y.shape[-1]}'
    avg_hr = 0
    total_positives = np.zeros(3)
    avg_hits = np.zeros(3)

    for i in range(y.shape[0]):
        input_idx = np.where(x[i, :] == 1)[0]
        score = y[i, :]

        viewed_item = set(input_idx)
        # print('LC > viewed_item:',viewed_item)
        positive_items = set(pos[i])
        negative_items = neg[i]
        neg_scores = sorted(score[negative_items].tolist(), reverse=True)

        # Tutti i positivi predetti meno quelli visti
        predicted_item = positive_items - viewed_item
        hit = 0
        hit_pop = [0, 0, 0]
        hit_pop_tot = [0, 0, 0]

        for pos_item in predicted_item:
            score_pos = score[pos_item]

            score_top_k = neg_scores[top_k - 1]

            if score_pos > score_top_k:
                hit += 1

            # popularity
            current_popularity = popularity[pos_item]

            if current_popularity <= popularity_thresholds[0]:

                hit_pop_tot[0] += 1
                if score_pos > score_top_k:
                    hit_pop[0] += 1

            elif popularity_thresholds[0] < current_popularity <= popularity_thresholds[1]:

                hit_pop_tot[1] += 1
                if score_pos > score_top_k:
                    hit_pop[1] += 1

            else:  # current_popularity > popularity_thresholds[1]

                hit_pop_tot[2] += 1
                if score_pos > score_top_k:
                    hit_pop[2] += 1

        assert hit <= len(predicted_item), f'{hit} / {len(predicted_item)}'
        assert sum(hit_pop) == hit, f'hit count error {hit} != {hit_pop}'
        assert len(predicted_item) == sum(hit_pop_tot), f'hit count error {len(predicted_item)} != {hit_pop_tot}'
        # avg_hr += hit / len(predicted_item)
        # avg_hits += np.array([a / b if a > 0 else 0 for a, b in zip(hit_pop, hit_pop_tot)])
        avg_hr += hit
        avg_hits += np.array(hit_pop)
        total_positives += np.array(hit_pop_tot)

    return avg_hr, avg_hits, total_positives


from torch.utils.checkpoint import checkpoint_sequential, checkpoint


class ModuleWrapperIgnores2ndArg(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x, dummy_arg=None):
        assert dummy_arg is not None
        x = self.module(x)
        return x


class MultiVAE(nn.Module):
    # https://discuss.pytorch.org/t/checkpoint-with-no-grad-requiring-inputs-problem/19117/11
    """
    Container module for Multi-VAE.
    Multi-VAE : Variational Autoencoder with Multinomial Likelihood
    See Variational Autoencoders for Collaborative Filtering
    https://arxiv.org/abs/1802.05814
    """

    def __init__(self, p_dims, q_dims=None, dropout=0.5, use_checkpoint=True, chunks=3):
        super(MultiVAE, self).__init__()

        self.use_checkpoint = use_checkpoint
        self.chunks = chunks

        self.p_dims = p_dims
        if q_dims:
            assert q_dims[0] == p_dims[-1], "In and Out dimensions must equal to each other"
            assert q_dims[-1] == p_dims[0], "Latent dimension for p- and q- network mismatches."
            self.q_dims = q_dims
        else:
            self.q_dims = p_dims[::-1]

        print('Encoder dimensions:', self.q_dims)
        print('Decoder dimensions:', self.p_dims)

        # Last dimension of q- network is for mean and variance

        temp_q_dims = self.q_dims[:-1] + [self.q_dims[-1] * 2]
        '''
        self.q_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                                       d_in, d_out in zip(temp_q_dims[:-1], temp_q_dims[1:])])
        self.p_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                                       d_in, d_out in zip(self.p_dims[:-1], self.p_dims[1:])])
        '''

        q_list = []
        idx = 1
        for d_in, d_out in zip(temp_q_dims[:-1], temp_q_dims[1:]):
            q_list.append(nn.Linear(d_in, d_out))
            print('idx:', idx)
            print('self.q_dims:', self.q_dims)
            if idx < len(self.q_dims) - 1:
                q_list.append(nn.ReLU())
            idx += 1
        self.q_layers = nn.Sequential(*q_list)

        p_list = []
        idx = 1
        for d_in, d_out in zip(self.p_dims[:-1], self.p_dims[1:]):
            p_list.append(nn.Linear(d_in, d_out))
            print('idx:', idx)
            print('self.p_dims:', self.p_dims)
            if idx < len(self.p_dims) - 1:
                p_list.append(nn.ReLU())
            idx += 1
        self.p_layers = nn.Sequential(*p_list)

        self.bn_enc = nn.ModuleList([nn.BatchNorm1d(d_out) for d_out in temp_q_dims[1:-1]])
        self.bn_dec = nn.ModuleList([nn.BatchNorm1d(d_out) for d_out in self.p_dims[1:-1]])

        self.drop = nn.Dropout(dropout)
        self.init_weights()

        # -------------------------------------------------

        self.dummy_tensor_q = torch.ones(1, dtype=torch.float32, requires_grad=True)
        self.module_wrapper_q = ModuleWrapperIgnores2ndArg(self.q_layers)

        self.dummy_tensor_p = torch.ones(1, dtype=torch.float32, requires_grad=True)
        self.module_wrapper_p = ModuleWrapperIgnores2ndArg(self.p_layers)

    def forward(self, input_data, predict=False):

        mu, logvar = self.encode(input_data)
        if predict:
            return self.decode(mu), mu, logvar

        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def encode(self, input_data):
        # h = F.normalize(input_data)
        # h = self.drop(h)

        # input_data.requires_grad_(True)
        # print('REQUIRES GRAD:', input_data.requires_grad)

        h = input_data
        if self.use_checkpoint:
            # h = checkpoint_sequential(self.q_layers, self.chunks, h)
            h = checkpoint(self.module_wrapper_q, h, self.dummy_tensor_q)
        else:
            h = self.q_layers(h)
        mu = h[:, :self.q_dims[-1]]
        logvar = h[:, self.q_dims[-1]:]
        return mu, logvar

    def decode(self, z):
        # z.requires_grad_(True)
        # print('REQUIRES GRAD:', z.requires_grad)

        if self.use_checkpoint:
            # return checkpoint_sequential(self.p_layers, self.chunks, z)
            return checkpoint(self.module_wrapper_p, z, self.dummy_tensor_p)
        else:
            return self.p_layers(z)

    def reparameterize(self, mu, logvar):
        # if self.training:
        #    std = torch.exp(0.5 * logvar)
        #    eps = torch.randn_like(std)
        #    return eps.mul(std).add_(mu)
        # else:
        #    return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def init_weights(self):
        for layer in self.q_layers:
            # Xavier Initialization for weights
            if isinstance(layer, nn.Linear):
                print('Xavier Inizialization for Layer:', layer)
                size = layer.weight.size()
                fan_out = size[0]
                fan_in = size[1]
                std = np.sqrt(2.0 / (fan_in + fan_out))
                layer.weight.data.normal_(0.0, std)

                # Normal Initialization for Biases
                layer.bias.data.normal_(0.0, 0.001)

        for layer in self.p_layers:
            # Xavier Initialization for weights
            if isinstance(layer, nn.Linear):
                print('Xavier Inizialization for Layer:', layer)
                size = layer.weight.size()
                fan_out = size[0]
                fan_in = size[1]
                std = np.sqrt(2.0 / (fan_in + fan_out))
                layer.weight.data.normal_(0.0, std)

                # Normal Initialization for Biases
                layer.bias.data.normal_(0.0, 0.001)


class rvae_loss(nn.Module):
    def __init__(self, popularity=None, scale=1., beta=1., thresholds=None):
        super(rvae_loss, self).__init__()

        if popularity is not None:
            # FIXME: make sure that keys are aligned with positions
            # self.popularity = torch.tensor(list(popularity.values())).to(device)
            self.popularity = torch.tensor(popularity).to(device)
            self.thresholds = thresholds
        else:
            self.popularity = None
            self.thresholds = None

        self.logsigmoid = torch.nn.LogSigmoid()

        self.scale = scale
        self.beta = beta

    def weight(self, pos_items, mask, one_as_default=True):
        if self.popularity is None:
            #            weight = 1 if one_as_default else mask
            weight = mask
        else:
            x = self.popularity[pos_items.long()]
            weight = ((x ** self.beta) * (2 - self.scale) + self.scale) / (1 + x ** self.beta) * mask

        # TODO
        weight = mask
        return weight

    def kld(self, mu, logvar):
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

        return KLD

    def forward(self, x, y, mu, logvar, anneal, **args):
        n_llk = self.log_p(x, y, **args)

        loss = n_llk + anneal * self.kld(mu, logvar)

        return loss


class rvae_rank_pair_loss(rvae_loss):
    def __init__(self, **kargs):
        super(rvae_rank_pair_loss, self).__init__(**kargs)

    def log_p(self, x, y, pos_items, neg_items, mask):
        ones = (mask[0, :] != 0).sum(dim=0).item()

        weight = self.weight(pos_items, mask, False)

        eps = 1e-10

        y1 = torch.gather(y, 1, (pos_items).long()) * mask
        y2 = torch.gather(y, 1, (neg_items).long()) * mask

        neg_ll = -torch.sum(self.logsigmoid(y1 - y2) * weight) / mask.sum()

        # del pos_items
        # del neg_items
        # del mask
        # del y1
        # del y2
        # torch.cuda.empty_cache()

        return neg_ll


class rvae_focal_loss(rvae_loss):
    def __init__(self, gamma=3, **kargs):
        super(rvae_focal_loss, self).__init__(**kargs)

        self.gamma = gamma
        self.sigmoid = torch.nn.Sigmoid()

    def log_p(self, x, y, pos_items, neg_items, mask):
        weight = self.weight(pos_items, mask)

        # log_p = torch.fun.sigmoid(weight*score)
        y1 = torch.gather(y, 1, (pos_items).long())
        y2 = torch.gather(y, 1, (neg_items).long())
        p = self.sigmoid(y1 - y2) * mask
        w = weight / self.scale * p
        w = w * (1 - p).pow(self.gamma)

        neg_ll = - w * self.logsigmoid(y1 - y2) * mask

        # neg_ll = - torch.pow(1 - torch.exp(log_p), self.beta) * weight * log_p

        neg_ll = torch.sum(neg_ll) / mask.sum()

        return neg_ll


"""# Train and test"""

trainloader = DataLoader(dataset_file, use_popularity=True)
# dataloader = DataLoaderDummy(None)
n_items = trainloader.n_items

# SETTINGS
settings = types.SimpleNamespace()
settings.dataset_name = os.path.split(data_dir)[-1]
settings.p_dims = [200, 600, n_items]
# settings.p_dims = [1, 5, 10, 50, 100, 200, 600, n_items]
settings.batch_size = 512
settings.weight_decay = 0.0
settings.learning_rate = 1e-3
# the total number of gradient updates for annealing
settings.total_anneal_steps = 200000
# largest annealing parameter
settings.anneal_cap = 0.2
settings.sample_mask = 100

settings.gamma_k = 1000

settings.metrics_alpha = 100
settings.metrics_beta = .03
settings.metrics_gamma = 5
settings.metrics_scale = 1 / 15
settings.metrics_percentile = .45

popularity = trainloader.item_popularity
thresholds = trainloader.thresholds

max_y_aux_popularity = compute_max_y_aux_popularity()

"""*testo in corsivo*# Check that ids are aligned
for i,x in enumerate(popularity.keys()):
    if i != x:
        print(f'warning: key {x} is position {i}')
"""


def naive_sparse2tensor(data):
    return torch.FloatTensor(data.toarray() if hasattr(data, 'toarray') else data)


def train(dataloader, epoch, optimizer):
    global update_count

    log_interval = int(trainloader.n_users * .7 / settings.batch_size // 4)
    if log_interval == 0:
        log_interval = 1

    # Turn on training mode
    model.train()
    train_loss = 0.0
    train_loss_cumulative = 0.0
    start_time = time.time()

    if epoch == 1:
        print(f'log every {log_interval} log interval')
        print(f'batches are {dataloader.n_items // settings.batch_size} with size {settings.batch_size}')

    for batch_idx, (x, pos, neg, mask) in enumerate(dataloader.iter(batch_size=settings.batch_size)):
        x = naive_sparse2tensor(x).to(device)
        pos_items = naive_sparse2tensor(pos).to(device)
        neg_items = naive_sparse2tensor(neg).to(device)
        mask = naive_sparse2tensor(mask).to(device)

        update_count += 1
        if settings.total_anneal_steps > 0:
            anneal = min(settings.anneal_cap, 1. * update_count / settings.total_anneal_steps)
        else:
            anneal = settings.anneal_cap

        # TRAIN on batch
        optimizer.zero_grad()
        y, mu, logvar = model(x)

        # loss = criterion(recon_batch, x, pos_items, neg_items, mask, mask, mu, logvar, anneal)
        loss = criterion(x, y, mu, logvar, anneal, pos_items=pos_items, neg_items=neg_items, mask=mask)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_loss_cumulative += loss.item()

        update_count += 1

        if batch_idx % log_interval == 0 and batch_idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:4d}/{:4d} batches | ms/batch {:4.2f} | loss {:4.4f}'.format(
                epoch, batch_idx, len(range(0, dataloader.size['train'], settings.batch_size)),
                elapsed * 1000 / log_interval,
                train_loss / log_interval))

            start_time = time.time()
            train_loss = 0.0

        if CUDA:
            torch.cuda.empty_cache()

    return train_loss_cumulative / (1 + batch_idx)


top_k = (1, 5, 10)


def evaluate(dataloader, normalized_popularity, tag='validation'):
    # Turn on evaluation mode
    model.eval()
    result = collections.defaultdict(float)
    batch_num = 0
    n_users_train = 0
    n_positives_predicted = np.zeros(3)
    result['train_loss'] = 0
    result['loss'] = 0

    with torch.no_grad():
        for batch_idx, (x, pos, neg, mask, pos_te, neg_te, mask_te) in enumerate(dataloader.iter_test(batch_size=settings.batch_size, tag=tag)):
            x_tensor = naive_sparse2tensor(x).to(device)
            pos = naive_sparse2tensor(pos).to(device)
            neg = naive_sparse2tensor(neg).to(device)
            mask = naive_sparse2tensor(mask).to(device)
            mask_te = naive_sparse2tensor(mask_te).to(device)

            batch_num += 1
            n_users_train += x_tensor.shape[0]

            x_input = x_tensor * (1 - mask_te)
            y, mu, logvar = model(x_input, True)

            #            loss = criterion(recon_batch, x_input, pos, neg, mask, mask, mu, logvar)
            loss = criterion(x_input, y, mu, logvar, 0, pos_items=pos, neg_items=neg, mask=mask)

            result['loss'] += loss.item()

            recon_batch_cpu = y.cpu().numpy()

            for k in top_k:
                out = compute_metric(x_input.cpu().numpy(),
                                     recon_batch_cpu,
                                     pos_te, neg_te,
                                     popularity,
                                     dataloader.thresholds,
                                     k)
                hitrate, popularity_hitrate, total_items = out

                result[f'hitrate@{k}'] += hitrate
                result[f'popularity_hitrate@{k}'] += popularity_hitrate
                # aggiornamento vettore del numero di items ripartiti per popolarita (l, m, h)
                if k == top_k[0]:
                    n_positives_predicted += total_items

    # last metric is str
    # n_users_pop = dataloader.get_users_counts_by_cat(tag)
    for i, k in enumerate(top_k):
        # result[f'hitrate@{k}'] = result[f'hitrate@{k}'] / n_users_train
        result[f'hitrate@{k}'] = result[f'hitrate@{k}'] / n_positives_predicted.sum()

        hits = result[f'popularity_hitrate@{k}']
        # result[f'popularity_hitrate@{k}'] = np.around(np.array(hits) / np.array(n_users_pop), 2)
        result[f'popularity_hitrate@{k}'] = np.around(np.array(hits) / n_positives_predicted, 2)
        result[f'popularity_hitrate@{k}'] = ', '.join([f'{x}' for x in result[f'popularity_hitrate@{k}']])

    return result


torch.set_printoptions(profile="full")

n_epochs = 200
update_count = 0
settings.batch_size = 1024  # 256
settings.learning_rate = 1e-3  # 1e-5
settings.optim = 'adam'
settings.scale = 1000
settings.use_popularity = True
settings.p_dims = [200, 600, n_items]
print(settings.p_dims)
model = MultiVAE(settings.p_dims)
model = model.to(device)
# summary(model, (3462,))

criterion = rvae_rank_pair_loss(popularity=popularity if settings.use_popularity else None,
                                scale=settings.scale,
                                thresholds=thresholds)
# criterion = rvae_focal_loss(popularity=popularity if settings.use_popularity else None, scale=settings.scale)

best_loss = np.Inf

stat_metric = []

print('At any point you can hit Ctrl + C to break out of training early.')
try:
    if settings.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=settings.learning_rate,
                                     weight_decay=settings.weight_decay)
    elif settings.optim == 'sgd':
        optimizer = torch.optim.SGD(params=model.parameters(), lr=settings.learning_rate, momentum=0,
                                    dampening=0, weight_decay=0, nesterov=False)
    else:
        optimizer = torch.optim.RMSprop(params=model.parameters(), lr=settings.learning_rate, alpha=0.99, eps=1e-08, weight_decay=settings.weight_decay, momentum=0, centered=False)

    for epoch in range(1, n_epochs + 1):
        epoch_start_time = time.time()
        train_loss = train(trainloader, epoch, optimizer)
        result = evaluate(trainloader, popularity)

        result['train_loss'] = train_loss
        stat_metric.append(result)

        print_metric = lambda k, v: f'{k}: {v:.4f}' if not isinstance(v, str) else f'{k}: {v}'
        ss = ' | '.join([print_metric(k, v) for k, v in stat_metric[-1].items() if k in ('train_loss', 'loss', 'hitrate@5', 'popularity_hitrate@5')])
        ss = f'| Epoch {epoch:3d} | time: {time.time() - epoch_start_time:4.2f}s | {ss} |'
        ls = len(ss)
        print('-' * ls)
        print(ss)
        print('-' * ls)

        # Save the model if the n100 is the best we've seen so far.

        print('best_loss:', best_loss)
        print('result[loss]:', result['loss'])

        if best_loss > result['loss']:  # and epoch>=20:
            print('saving model...')
            print(file_model)
            with open(file_model, 'wb') as f:
                torch.save(model, f)
            best_loss = result['loss']

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# output.show()

with open(file_model, 'rb') as f:
    model = torch.load(f)

"""# Training stats"""

# print(f'K = {settings.gamma_k}')
print('\n'.join([f'{k:<23}{v}' for k, v in sorted(stat_metric[-1].items())]))

# LOSS
lossTrain = [x['train_loss'] for x in stat_metric]
lossTest = [x['loss'] for x in stat_metric]

lastHitRate = [x['hitrate@5'] for x in stat_metric]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 8))
ax1.plot(lossTrain, color='b', )
# ax1.set_yscale('log')
ax1.set_title('Train')

ax2.plot(lossTest, color='r')
# ax2.set_yscale('log')
ax2.set_title('Validation')

ax3.plot(lastHitRate)
ax3.set_title('HitRate@5');

fig, axes = plt.subplots(4, 3, figsize=(20, 20))

axes = axes.ravel()
i = 0

for k in top_k:
    hitRate = [x[f'hitrate@{k}'] for x in stat_metric]

    ax = axes[i]
    i += 1

    ax.plot(hitRate)
    ax.set_title(f'HitRate@{k}')

for j, name in enumerate('LessPop MiddlePop TopPop'.split()):
    for k in top_k:
        hitRate = [float(x[f'popularity_hitrate@{k}'].split(',')[j]) for x in stat_metric]

        ax = axes[i]
        i += 1

        ax.plot(hitRate)
        ax.set_title(f'{name} popularity_hitrate@{k}')

plt.show();

"""# Test stats"""

model.eval()
result_test = evaluate(trainloader, popularity, 'test')

print(f'K = {settings.gamma_k}')
print('\n'.join([f'{k:<23}{v}' for k, v in sorted(result_test.items())]))

"""# Save result"""

rundate = datetime.today().strftime('%Y%m%d_%H%M')
lossname = criterion.__class__.__name__

folder_name = os.path.join(result_dir, f'run_{settings.dataset_name}_{rundate}_{lossname}')
os.mkdir(folder_name)

# result info
with open(os.path.join(folder_name, 'info.txt'), 'w') as fp:
    fp.write(f'K = {settings.gamma_k}\n')
    fp.write(f'Loss = {lossname}\n')
    fp.write(f'Epochs train = {len(stat_metric)}\n')

    fp.write('\n')

    for k in dir(settings):
        if not k.startswith('__'):
            fp.write(f'{k} = {getattr(settings, k)}\n')

#    fp.write('\n' * 4)
#    model_print, _ = torchsummary.summary_string(model, (dataloader.n_items,), device='gpu' if CUDA else 'cpu')
#    fp.write(model_print)


# all results
with open(os.path.join(folder_name, 'result.json'), 'w') as fp:
    json.dump(stat_metric, fp)

# test results
with open(os.path.join(folder_name, 'result_test.json'), 'w') as fp:
    json.dump(result_test, fp, indent=4, sort_keys=True)

# chart 1
lossTrain = [x['train_loss'] for x in stat_metric]
lossTest = [x['loss'] for x in stat_metric]

lastHitRate = [x['hitrate@5'] for x in stat_metric]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 8))
ax1.plot(lossTrain, color='b', )
# ax1.set_yscale('log')
ax1.set_title('Train')

ax2.plot(lossTest, color='r')
# ax2.set_yscale('log')
ax2.set_title('Validation')

ax3.plot(lastHitRate)
ax3.set_title('HitRate@5')

plt.savefig(os.path.join(folder_name, 'loss.png'));

# chart 2
fig, axes = plt.subplots(4, 3, figsize=(20, 20))

axes = axes.ravel()
i = 0

for k in top_k:
    hitRate = [x[f'hitrate@{k}'] for x in stat_metric]

    ax = axes[i]
    i += 1

    ax.plot(hitRate)
    ax.set_title(f'HitRate@{k}')

for j, name in enumerate('LessPop MiddlePop TopPop'.split()):
    for k in top_k:
        hitRate = [float(x[f'popularity_hitrate@{k}'].split(',')[j]) for x in stat_metric]

        ax = axes[i]
        i += 1

        ax.plot(hitRate)
        ax.set_title(f'{name} popularity_hitrate@{k}')

plt.savefig(os.path.join(folder_name, 'hr.png'));

print('DONE', folder_name)
