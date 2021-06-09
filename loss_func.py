import torch
import torch.nn as nn


class rvae_loss(nn.Module):
    def __init__(self, popularity=None, scale=1., beta=1., thresholds=None, frequencies=None, device="cpu"):
        super(rvae_loss, self).__init__()

        if popularity is not None:
            # FIXME: make sure that keys are aligned with positions
            # self.popularity = torch.tensor(list(popularity.values())).to(device)
            self.popularity = torch.tensor(popularity).to(device)
            self.frequencies = torch.tensor(frequencies).to(device)
            self.thresholds = thresholds
        else:
            self.popularity = None
            self.thresholds = None

        self.logsigmoid = torch.nn.LogSigmoid()

        self.scale = scale
        self.beta = beta

    def weight(self, pos_items, mask, one_as_default=True):
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
        # loss = n_llk + self.kld(mu, logvar)

        return loss


class ensemble_rvae_loss(nn.Module):
    def __init__(self, popularity=None, scale=1., beta=1., thresholds=None, frequencies=None, device="cpu"):
        super(ensemble_rvae_loss, self).__init__()

        if popularity is not None:
            # FIXME: make sure that keys are aligned with positions
            # self.popularity = torch.tensor(list(popularity.values())).to(device)
            self.popularity = torch.tensor(popularity).to(device)
            self.frequencies = torch.tensor(frequencies).to(device)
            self.thresholds = thresholds
        else:
            self.popularity = None
            self.thresholds = None

        self.logsigmoid = torch.nn.LogSigmoid()

        self.scale = scale
        self.beta = beta

    def weight(self, pos_items, mask, one_as_default=True):
        weight = mask
        return weight

    def kld(self, mu, logvar):
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

        return KLD


class rvae_rank_pair_loss(rvae_loss):
    def __init__(self, device, **kargs):
        super(rvae_rank_pair_loss, self).__init__(**kargs)
        self.device = device

    def log_p(self, x, y, pos_items, neg_items, mask, BASELINE):
        weight = mask

        y1 = torch.gather(y, 1, (pos_items).long()) * mask
        y2 = torch.gather(y, 1, (neg_items).long()) * mask

        pop_pos = self.popularity[pos_items.long()]
        pop_neg = self.popularity[neg_items.long()]

        filter_pos = (pop_pos <= self.thresholds[0]).float().to(self.device)  # low
        filter_neg = (pop_neg > self.thresholds[0]).float()  # low

        # freq_pos = self.frequencies[pos_items.long()].float()
        # freq_neg = self.frequencies[neg_items.long()].float()

        if BASELINE:
            neg_ll = - torch.sum(self.logsigmoid(y1 - y2) * weight) / mask.sum()
        else:
            neg_ll = - torch.sum(filter_pos * self.logsigmoid(y1 - y2) * weight) / mask.sum()
            # neg_ll = - torch.sum(self.logsigmoid((1 - pop_pos) * y1 - y2) * weight) / mask.sum()
            # neg_ll = - torch.sum(filter_pos * filter_neg*self.logsigmoid(y1 - y2) * weight) / mask.sum()

        del pop_pos
        del pop_neg
        del filter_pos
        del filter_neg

        torch.cuda.empty_cache()

        return neg_ll


class ensemble_rvae_rank_pair_loss(rvae_loss):
    def __init__(self, **kargs):
        super(ensemble_rvae_rank_pair_loss, self).__init__(**kargs)

    def log_p(self, x, y, pos_items, neg_items, mask):
        '''
        weight = mask

        y1 = torch.gather(y, 1, (pos_items).long()) * mask
        y2 = torch.gather(y, 1, (neg_items).long()) * mask

        neg_ll = - torch.sum(self.logsigmoid(y1 - y2) * weight) / mask.sum()

        del y1
        del y2
        torch.cuda.empty_cache()
        '''
        return 0
        # return neg_ll


class rvae_focal_loss(rvae_loss):
    def __init__(self, gamma=3, **kargs):
        super(rvae_focal_loss, self).__init__(**kargs)

        self.gamma = gamma
        self.sigmoid = torch.nn.Sigmoid()

    def log_p(self, x, y, pos_items, neg_items, mask):
        weight = self.weight(pos_items, mask)

        # y1 = torch.gather(y, 1, (pos_items).long()) * mask
        # y2 = torch.gather(y, 1, (neg_items).long()) * mask
        # neg_ll = - torch.sum(self.logsigmoid(y1 - y2)*weight) / mask.sum()
        # neg_ll = - torch.pow(1 - torch.exp(neg_ll), 5) * neg_ll

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


class ensemble_rvae_focal_loss(ensemble_rvae_loss):
    def __init__(self, gamma=3, **kargs):
        super(ensemble_rvae_focal_loss, self).__init__(**kargs)

        self.gamma = gamma
        self.sigmoid = torch.nn.Sigmoid()

    def log_p(self, x, y, pos_items, neg_items, mask):
        weight = self.weight(pos_items, mask)

        # y1 = torch.gather(y, 1, (pos_items).long()) * mask
        # y2 = torch.gather(y, 1, (neg_items).long()) * mask
        # neg_ll = - torch.sum(self.logsigmoid(y1 - y2)*weight) / mask.sum()
        # neg_ll = - torch.pow(1 - torch.exp(neg_ll), 5) * neg_ll

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
