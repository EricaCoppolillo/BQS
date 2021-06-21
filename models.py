import torch
import torch.nn as nn
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


class EnsembleMultiVAE(nn.Module):

    def __init__(self, n_items, popularity, thresholds=None, gamma=.4, device="cpu"):
        super(EnsembleMultiVAE, self).__init__()

        self.n_items = n_items
        self.test_print = False
        self.popularity = popularity
        self.thresholds = thresholds
        self.gamma = gamma

        self.filter_a = torch.tensor(np.array(self.popularity) > self.thresholds[0]).to(device).float()  # baseline
        self.filter_b = torch.tensor(np.array(self.popularity) <= self.thresholds[0]).to(device).float()  # low

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

        if self.test_print:
            print('ENSEMBLE TEST PRINT (train) ---------------------------------------')
            print('Shape x, x[0]:', len(x), len(x[0]))
            print('Shape y_a, y_a[0]:', len(y_a), len(y_a[0]))
            print('Shape y_b, y_b[0]:', len(y_b), len(y_b[0]))
            print('Shape popularity:', len(self.popularity))
            print('Type popularity:', type(self.popularity))
            print('x[0][:100]:', x[0][:100])
            print('y_a[0][:100]:', y_a[0][:100])
            print('y_b[0][:100]:', y_b[0][:100])
            print('z_a[0][:100]:', z_a[0][:100])
            print('z_b[0][:100]:', z_b[0][:100])
            print('popularity:', self.popularity[:100])
            print('filter a:', self.filter_a[:100])
            print('filter b:', self.filter_b[:100])
            print('thresholds:', self.thresholds)
            print('-------------------------------------------------------------------')
            self.test_print = False

        baseline = False

        if baseline:
            y_e = z_a
        else:
            # y_e = z_a * self.filter_a + z_b * self.filter_b * gamma
            y_e = z_a + z_b * self.gamma

        '''
        CITE U LIKE ----------------------------------------------------------------------------------------------------

        BASELINE
        luciano_stat_by_pop@1  0.07,0.22,0.67
        luciano_stat_by_pop@10 0.48,0.75,0.95
        luciano_stat_by_pop@5  0.29,0.57,0.90

        ENSEMBLE (old)
        luciano_stat_by_pop@1  0.17,0.20,0.66
        luciano_stat_by_pop@10 0.72,0.63,0.91
        luciano_stat_by_pop@5  0.51,0.49,0.87

        ENSEMBLE (filter gamma=1.0)
        luciano_stat_by_pop@1  0.16,0.19,0.65
        luciano_stat_by_pop@10 0.76,0.53,0.88
        luciano_stat_by_pop@5  0.52,0.43,0.84

        ENSEMBLE (filter gamma=0.9)
        luciano_stat_by_pop@1  0.15,0.20,0.66
        luciano_stat_by_pop@10 0.71,0.57,0.89
        luciano_stat_by_pop@5  0.47,0.46,0.85

        ENSEMBLE (filter gamma=0.5)
        luciano_stat_by_pop@1  0.09,0.21,0.66
        luciano_stat_by_pop@10 0.60,0.63,0.91
        luciano_stat_by_pop@5  0.38,0.50,0.87
        '''

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
