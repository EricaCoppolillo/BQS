import torch
import torch.nn as nn
from config import Config

model_types = Config("./model_type_info.json")


class bpr_loss(nn.Module):
    def __init__(self, is_weighted_model):
        super(bpr_loss, self).__init__()
        self.is_weighted_model = is_weighted_model
        self.logsigmoid = torch.nn.LogSigmoid()

    def forward(self, output_scores, mask):
        weight = mask
        if self.is_weighted_model:
            mask = mask > 0

        loss_value = - (self.logsigmoid(output_scores) * weight) / mask.sum()
        return loss_value




class bpr_rank_pair_loss(bpr_loss):
    def __init__(self, device, **kargs):
        super(bpr_rank_pair_loss, self).__init__(**kargs)
        self.device = device

    def log_p(self, x, y, pos_items, neg_items, mask, model_type):

        # assert mask.sum() > 0
        if model_type == model_types.REWEIGHTING:
            weight = mask
            # assert weight.sum() > 0
            mask = mask > 0
        else:
            weight = mask

        y1 = torch.gather(y, 1, pos_items.long()) * mask
        y2 = torch.gather(y, 1, neg_items.long()) * mask

        # Building filters for different classes of items
        if model_type == model_types.MED:
            pop_pos = self.popularity[pos_items.long()]
            filter_pos = (self.thresholds[0] < pop_pos <= self.thresholds[1]).float().to(self.device)  # medium
        elif model_type == model_types.HIGH:
            pop_pos = self.popularity[pos_items.long()]
            filter_pos = (pop_pos > self.thresholds[1]).float().to(self.device)  # high
        elif model_type == model_types.LOW:
            pop_pos = self.popularity[pos_items.long()]
            filter_pos = (pop_pos <= self.thresholds[0]).float().to(self.device)  # low

        if model_type in (model_types.BASELINE, model_types.REWEIGHTING, model_types.OVERSAMPLING):
            neg_ll = - (self.logsigmoid(y1 - y2) * weight) / mask.sum()
        else:
            neg_ll = - (filter_pos * self.logsigmoid(y1 - y2) * weight) / mask.sum()
            del pop_pos
            del filter_pos

        torch.cuda.empty_cache()
        return neg_ll


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
        n_llk = torch.sum(self.log_p(x, y, **args))
        loss = n_llk + anneal * self.kld(mu, logvar)

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


class rvae_rank_pair_loss(rvae_loss):
    def __init__(self, device, **kargs):
        super(rvae_rank_pair_loss, self).__init__(**kargs)
        self.device = device

    def log_p(self, x, y, pos_items, neg_items, mask, model_type):

        # assert mask.sum() > 0
        if model_type == model_types.REWEIGHTING:
            weight = mask
            # assert weight.sum() > 0
            mask = mask > 0
        else:
            weight = mask

        y1 = torch.gather(y, 1, pos_items.long()) * mask
        y2 = torch.gather(y, 1, neg_items.long()) * mask

        # Building filters for different classes of items
        if model_type == model_types.MED:
            pop_pos = self.popularity[pos_items.long()]
            filter_pos = (self.thresholds[0] < pop_pos <= self.thresholds[1]).float().to(self.device)  # medium
        elif model_type == model_types.HIGH:
            pop_pos = self.popularity[pos_items.long()]
            filter_pos = (pop_pos > self.thresholds[1]).float().to(self.device)  # high
        elif model_type == model_types.LOW:
            pop_pos = self.popularity[pos_items.long()]
            filter_pos = (pop_pos <= self.thresholds[0]).float().to(self.device)  # low

        if model_type in (model_types.BASELINE, model_types.REWEIGHTING, model_types.OVERSAMPLING, model_types.U_SAMPLING):
            # assert mask.sum() > 0
            neg_ll = - (self.logsigmoid(y1 - y2) * weight) / mask.sum()
        else:
            neg_ll = - (filter_pos * self.logsigmoid(y1 - y2) * weight) / mask.sum()
            del pop_pos
            del filter_pos

        torch.cuda.empty_cache()
        return neg_ll


class vae_loss(rvae_loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.bce = torch.nn.BCEWithLogitsLoss()

    def log_p(self, x, y, **kwargs):
        neg_ll = self.bce(y, x)
        return neg_ll


class ensemble_rvae_rank_pair_loss(rvae_loss):
    def __init__(self, **kargs):
        super(ensemble_rvae_rank_pair_loss, self).__init__(**kargs)

    def log_p(self, x, y, pos_items, neg_items, mask, model_type=model_types.BASELINE):

        # assert mask.sum() > 0
        if model_type == model_types.REWEIGHTING:
            weight = mask
            # assert weight.sum() > 0
            mask = mask > 0
        else:
            weight = mask

        y1 = torch.gather(y, 1, pos_items.long()) * mask
        y2 = torch.gather(y, 1, neg_items.long()) * mask

        # Building filters for different classes of items
        if model_type == model_types.MED:
            pop_pos = self.popularity[pos_items.long()]
            filter_pos = (self.thresholds[0] < pop_pos <= self.thresholds[1]).float().to(self.device)  # medium
        elif model_type == model_types.HIGH:
            pop_pos = self.popularity[pos_items.long()]
            filter_pos = (pop_pos > self.thresholds[1]).float().to(self.device)  # high
        elif model_type == model_types.LOW:
            pop_pos = self.popularity[pos_items.long()]
            filter_pos = (pop_pos <= self.thresholds[0]).float().to(self.device)  # low

        if model_type in (model_types.BASELINE, model_types.REWEIGHTING, model_types.OVERSAMPLING):
            # assert mask.sum() > 0
            neg_ll = - torch.sum(self.logsigmoid(y1 - y2) * weight) / mask.sum()
        else:
            neg_ll = - torch.sum(filter_pos * self.logsigmoid(y1 - y2) * weight) / mask.sum()
            del pop_pos
            del filter_pos

        torch.cuda.empty_cache()
        return neg_ll

    def forward(self, x, y, **args):
        n_llk = self.log_p(x, y, **args)

        return n_llk


class rvae_focal_loss(rvae_loss):
    def __init__(self, gamma=3, **kargs):
        super(rvae_focal_loss, self).__init__(**kargs)

        self.gamma = gamma
        self.sigmoid = torch.nn.Sigmoid()

    def log_p(self, x, y, pos_items, neg_items, mask):
        weight = self.weight(pos_items, mask)

        y1 = torch.gather(y, 1, pos_items.long())
        y2 = torch.gather(y, 1, neg_items.long())
        p = self.sigmoid(y1 - y2) * mask
        w = weight / self.scale * p
        w = w * (1 - p).pow(self.gamma)

        neg_ll = - w * self.logsigmoid(y1 - y2) * mask
        neg_ll = torch.sum(neg_ll) / mask.sum()

        return neg_ll


class ensemble_rvae_focal_loss(ensemble_rvae_loss):
    def __init__(self, gamma=3, **kargs):
        super(ensemble_rvae_focal_loss, self).__init__(**kargs)

        self.gamma = gamma
        self.sigmoid = torch.nn.Sigmoid()

    def log_p(self, x, y, pos_items, neg_items, mask):
        weight = self.weight(pos_items, mask)

        y1 = torch.gather(y, 1, pos_items.long())
        y2 = torch.gather(y, 1, neg_items.long())
        p = self.sigmoid(y1 - y2) * mask
        w = weight / self.scale * p
        w = w * (1 - p).pow(self.gamma)

        neg_ll = - w * self.logsigmoid(y1 - y2) * mask
        neg_ll = torch.sum(neg_ll) / mask.sum()
        return neg_ll


