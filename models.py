import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MultiVAE(nn.Module):
    """
    Container module for Multi-VAE.
    Multi-VAE : Variational Autoencoder with Multinomial Likelihood
    See Variational Autoencoders for Collaborative Filtering
    https://arxiv.org/abs/1802.05814
    """

    def __init__(self, p_dims, q_dims=None, dropout=0.5):
        super(MultiVAE, self).__init__()
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
        self.q_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                                       d_in, d_out in zip(temp_q_dims[:-1], temp_q_dims[1:])])
        self.p_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                                       d_in, d_out in zip(self.p_dims[:-1], self.p_dims[1:])])

        self.bn_enc = nn.ModuleList([nn.BatchNorm1d(d_out) for d_out in temp_q_dims[1:-1]])
        self.bn_dec = nn.ModuleList([nn.BatchNorm1d(d_out) for d_out in self.p_dims[1:-1]])

        self.drop = nn.Dropout(dropout)
        self.init_weights()

    def forward(self, input_data, predict=False):
        mu, logvar = self.encode(input_data)
        if predict:
            return self.decode(mu), mu, logvar

        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def encode(self, input_data):
        # h = F.normalize(input_data)
        # h = self.drop(h)
        h = input_data
        mu, logvar = None, None
        for i, layer in enumerate(self.q_layers):
            h = layer(h)
            if i != len(self.q_layers) - 1:
                # h = self.bn_enc[i](h)
                # h = torch.tanh(h)
                h = torch.relu(h)
            else:
                mu = h[:, :self.q_dims[-1]]
                logvar = h[:, self.q_dims[-1]:]
        return mu, logvar

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

    def decode(self, z):
        h = z
        for i, layer in enumerate(self.p_layers):
            h = layer(h)
            if i != len(self.p_layers) - 1:
                # h = torch.tanh(h)
                # h = self.bn_dec[i](h)
                h = torch.relu(h)
        return h

    def init_weights(self):
        for layer in self.q_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)

        for layer in self.p_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)


class EnsembleMultiVAENet(nn.Module):

    def __init__(self, n_items):
        super().__init__()

        # self.layer = nn.Linear(2 * n_items, n_items)
        # self.layer = nn.Linear(2, 1)
        a = torch.ones(size=(n_items,)) * .5
        self.alpha = nn.Parameter(a)
        self.beta = 0

    def forward(self, x, y_a, y_b):
        print(x.shape)
        z_a = torch.softmax(y_a, 1)
        z_b = torch.softmax(y_b, 1)
        # z_a = z_a.unsqueeze(-1)
        # z_b = z_b.unsqueeze(-1)
        #
        # y_e = torch.cat((z_a, z_b), dim=-1)
        # y_e = self.layer(y_e)
        #
        # y_e = y_e.squeeze()

        alpha = torch.sigmoid(self.alpha)

        alpha = alpha.repeat(z_a.shape[0], 1)
        y_e = alpha * z_a + (1-alpha) * z_b

        return y_e


class EnsembleMultiVAETrainable(nn.Module):

    def __init__(self):  # n_items, popularity, thresholds=None, device='cpu'):
        super().__init__()

        # self.n_items = n_items
        # self.test_print = False
        # self.popularity = popularity
        # self.thresholds = thresholds

        self.alpha = nn.Parameter(torch.tensor(1.))
        self.beta = nn.Parameter(torch.tensor(1.))

        # self.beta = 0
        # self.alpha = nn.Parameter(torch.tensor(.5))

        # self.filter_a = torch.tensor(np.array(self.popularity) > self.thresholds[0]).to(device).float()  # baseline
        # self.filter_b = torch.tensor(np.array(self.popularity) <= self.thresholds[0]).to(device).float()  # low

    def forward(self, x, y_a, y_b):
        z_a = torch.softmax(y_a, 1)
        z_b = torch.softmax(y_b, 1)
        # z_a = y_a
        # z_b = y_b

        # y_e = self.alpha * z_a + self.beta * z_b
        # alpha = torch.sigmoid(self.alpha)
        # y_e = alpha * z_a + (1 - alpha) * z_b

        # alpha = torch.sigmoid(self.alpha)
        # beta = torch.sigmoid(self.beta)
        alpha = self.alpha
        beta = self.beta
        y_e = alpha * z_a + beta * z_b
        return y_e


class EnsembleMultiVAE(nn.Module):

    def __init__(self, n_items, popularity, thresholds=None, gamma=.4, device="cpu"):
        super(EnsembleMultiVAE, self).__init__()

        self.n_items = n_items
        self.test_print = False
        self.popularity = popularity
        self.thresholds = thresholds
        self.gamma = gamma

        # self.filter_a = torch.tensor(np.array(self.popularity) > self.thresholds[0]).to(device).float()  # baseline
        # self.filter_b = torch.tensor(np.array(self.popularity) <= self.thresholds[0]).to(device).float()  # low

        # low
        # self.mask = torch.tensor(np.array(self.popularity) <= self.thresholds[0], dtype=torch.int8)

    # def to(self, device):
    #     model = super().to(device)
    #     model.mask = model.mask.to(device)
    #     return model

    def normalize(self, tensor):
        min_v = torch.min(tensor)
        range_v = torch.max(tensor) - min_v
        if range_v > 0:
            normalised = (tensor - min_v) / range_v
        else:
            normalised = torch.zeros(tensor.size())
        return normalised

    def forward(self, x, y_a, y_b, predict=False):
        z_a = torch.softmax(y_a, 1)
        z_b = torch.softmax(y_b, 1)


        # if self.test_print:
        #     print('ENSEMBLE TEST PRINT (train) ---------------------------------------')
        #     print('Shape x, x[0]:', len(x), len(x[0]))
        #     print('Shape y_a, y_a[0]:', len(y_a), len(y_a[0]))
        #     print('Shape y_b, y_b[0]:', len(y_b), len(y_b[0]))
        #     print('Shape popularity:', len(self.popularity))
        #     print('Type popularity:', type(self.popularity))
        #     print('x[0][:100]:', x[0][:100])
        #     print('y_a[0][:100]:', y_a[0][:100])
        #     print('y_b[0][:100]:', y_b[0][:100])
        #     print('z_a[0][:100]:', z_a[0][:100])
        #     print('z_b[0][:100]:', z_b[0][:100])
        #     print('popularity:', self.popularity[:100])
        #     print('filter a:', self.filter_a[:100])
        #     print('filter b:', self.filter_b[:100])
        #     print('thresholds:', self.thresholds)
        #     print('-------------------------------------------------------------------')
        #     self.test_print = False

        # baseline = False

        # if baseline:
        #     y_e = z_a
        # else:
        #     # y_e = z_a * self.filter_a + z_b * self.filter_b * gamma

        y_e = z_a + z_b * self.gamma

        # y_e = (1-self.mask) * z_a + self.mask * z_b * self.gamma

        # y_e = z_a + self.mask * z_b * self.gamma
        
        return y_e

    def init_weights(self):
        for layer in self.layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)


class BPR(torch.nn.Module):
    def __init__(self, n_users, n_items, n_factors):
        super(BPR, self).__init__()

        self.embed_user = torch.nn.Embedding(n_users, n_factors)
        self.embed_item = torch.nn.Embedding(n_items, n_factors)

        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.embed_user.weight)
        torch.nn.init.xavier_uniform_(self.embed_item.weight)

    def forward(self, user, item_i, item_j=None):
        user = self.embed_user(user)
        item_i = self.embed_item(item_i)
        prediction_i = (user * item_i).sum(dim=-1)

        if item_j is not None:
            item_j = self.embed_item(item_j)
            prediction_j = (user * item_j).sum(dim=-1)

            return prediction_i - prediction_j

        return prediction_i

    def score(self, user, items):
        user = self.embed_user(user)
        user = user.unsqueeze(-1)

        items_t = self.embed_item(items)
        scores = torch.bmm(items_t, user).squeeze(-1)
        # scores = torch.mm(items_t, user).squeeze(-1)

        return scores


class BPR_MF(nn.Module):
    # provided by https://github.com/sh0416/bpr/blob/master/train.py
    def __init__(self, user_size, item_size, dim, weight_decay):
        super().__init__()
        self.W = nn.Parameter(torch.empty(user_size, dim))
        self.H = nn.Parameter(torch.empty(item_size, dim))
        nn.init.xavier_normal_(self.W.data)
        nn.init.xavier_normal_(self.H.data)
        self.weight_decay = weight_decay

    def forward(self, u, i, j):
        """Return loss value.

        Args:
            u(torch.LongTensor): tensor stored user indexes. [batch_size,]
            i(torch.LongTensor): tensor stored item indexes which is prefered by user. [batch_size,]
            j(torch.LongTensor): tensor stored item indexes which is not prefered by user. [batch_size,]

        Returns:
            torch.FloatTensor
        """
        u = self.W[u, :]
        i = self.H[i, :]
        j = self.H[j, :]
        x_ui = torch.mul(u, i).sum(dim=1)
        x_uj = torch.mul(u, j).sum(dim=1)
        x_uij = x_ui - x_uj
        log_prob = F.logsigmoid(x_uij).sum()
        regularization = self.weight_decay * (
                    u.norm(dim=1).pow(2).sum() + i.norm(dim=1).pow(2).sum() + j.norm(dim=1).pow(2).sum())
        return -log_prob + regularization

    def forward(self, u):
        """Return probability distribution on the set of items

        Args:
            u(torch.LongTensor): tensor stored user indexes. [batch_size,]
        Returns:
            torch.FloatTensor
        """
        u = self.W[u, :]
        x_ui = torch.mm(u, self.H.t())
        return x_ui

    def recommend(self, u):
        """Return recommended item list given users.
        Args:
            u(torch.LongTensor): tensor stored user indexes. [batch_size,]
        Returns:
            pred(torch.LongTensor): recommended item list sorted by preference. [batch_size, item_size]
        """
        u = self.W[u, :]
        x_ui = torch.mm(u, self.H.t())
        pred = torch.argsort(x_ui, dim=1)
        return pred