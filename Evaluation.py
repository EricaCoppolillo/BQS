from numpy import array
from tqdm import tqdm
from torch import arange as t_arange, argmax as t_argmax, cumsum as t_cumsum
from torch import empty_like as t_empty_like, gather as t_gather, log2 as t_log2
from torch import mean as t_mean, min as t_min, randint as t_randint, sort as t_sort
from torch import sum as t_sum, tensor
from torch.nn import Module
from torch.utils.data import DataLoader, TensorDataset
from sys import stdout


class _TinyModel(Module):

    def __init__(self, coeff=1):
        super(_TinyModel, self).__init__()
        self.coeff = coeff

    def forward(self, x):
        return self.coeff * x


def _compute_mrr(hit_matrix):
    idx = t_argmax(hit_matrix, 1)
    num = hit_matrix[t_arange(idx.size(0)), idx]

    return t_mean(num / (idx + 1))


def _compute_incremental_mrr(hit_matrix, y):
    recsize = hit_matrix.size(1)
    idx = t_argmax(hit_matrix, 1, keepdim=True)
    den = t_arange(recsize).unsqueeze(0).expand(idx.size(0), recsize) + 1
    vals = den[t_arange(den.size(0)), idx.squeeze()].unsqueeze(-1)
    den[den < vals] = 0
    den = (den / recsize).ceil() * vals
    ground, _ = y.max(dim=1, keepdim=True)
    den *= ground
    num = (t_cumsum(hit_matrix, dim=1) / recsize).ceil()

    return t_mean(((num / den).nan_to_num(nan=0) / ground).nan_to_num(nan=1), dim=0).numpy()


def _compute_ndcg(recsize, hit_matrix, limits):
    den = t_log2(t_arange(2, recsize + 2))
    dcg = t_sum(hit_matrix / den, dim=-1)

    num = t_arange(limits.max().item()).unsqueeze(0)
    num = num.expand(hit_matrix.size(0), num.size(1))
    num = (num < limits.unsqueeze(-1)) + 0
    idcg = t_sum(num / den[:num.size(1)], dim=-1)

    return t_mean((dcg / idcg).nan_to_num(nan=1))


def _compute_incremental_ndcg(recsize, hit_matrix, limits):
    den = t_log2(t_arange(2, recsize + 2))
    dcg = t_cumsum(hit_matrix / den, dim=-1)

    num = t_arange(limits.max().item()).unsqueeze(0)
    num = num.expand(hit_matrix.size(0), num.size(1))
    num = (num < limits[:, :num.size(1)]) + 0
    idcg = t_cumsum(num / den[:num.size(1)], dim=-1)

    if idcg.size(1) < dcg.size(1):
        tmp = t_empty_like(dcg)
        tmp[:, :idcg.size(1)] = idcg
        tmp[:, idcg.size(1):] = idcg[:, -1].unsqueeze(-1)
        idcg = tmp
        del tmp

    return t_mean((dcg / idcg).nan_to_num(nan=1), dim=0).numpy()


def _compute_precision(hits, limits):
    return t_mean((hits / limits).nan_to_num(nan=1))


def _compute_incremental_precision(hits, limits):
    return t_mean((hits / limits).nan_to_num(nan=1), dim=0).numpy()


def _compute_recall(hits, n_items_per_user):
    return t_mean((hits / n_items_per_user).nan_to_num(nan=1))


def _compute_incremental_recall(hits, n_items_per_user):
    return t_mean((hits / n_items_per_user).nan_to_num(nan=1), dim=0).numpy()


def _mymean(vec):
    return tensor(vec).mean().item()


def _my_incremental_mean(matrix):
    return array(matrix).mean(axis=0)


def evaluation(model: Module, testset: DataLoader, recsize: int):
    model.eval()
    precision, recall, mrr, ndcg = [], [], [], []

    for x, y in tqdm(testset, file=stdout):
        _, p_items = t_sort(model(x), descending=True)
        p_items = p_items[:, :recsize]
        hit_matrix = t_gather(y, dim=1, index=p_items)
        hits = t_sum(hit_matrix, dim=-1)
        n_items_per_user = t_sum(y, dim=-1)
        limits = t_min(tensor([recsize] * len(x)), n_items_per_user)

        precision.append(_compute_precision(hits, limits))
        recall.append(_compute_recall(hits, n_items_per_user))
        mrr.append(_compute_mrr(hit_matrix))
        ndcg.append(_compute_ndcg(recsize, hit_matrix, limits))

    result = {
        'precision': _mymean(precision),
        'recall': _mymean(recall),
        'mrr': _mymean(mrr),
        'ndcg': _mymean(ndcg)
    }

    return result


def incremental_evaluation(model: Module, testset: DataLoader, max_recsize: int):
    model.eval()
    precision, recall, mrr, ndcg = [], [], [], []

    for x, y in tqdm(testset, file=stdout):
        _, p_items = t_sort(model(x), descending=True)
        p_items = p_items[:, :max_recsize]
        hit_matrix = t_gather(y, dim=1, index=p_items)
        hits = t_cumsum(hit_matrix, dim=-1)
        n_items_per_user = t_sum(y, dim=-1).unsqueeze(-1)
        limits = t_min(t_arange(1, max_recsize + 1).unsqueeze(0).expand(
            x.size(0), max_recsize), n_items_per_user)

        precision.append(_compute_incremental_precision(hits, limits))
        recall.append(_compute_incremental_recall(hits, n_items_per_user))
        mrr.append(_compute_incremental_mrr(hit_matrix, y))
        ndcg.append(_compute_incremental_ndcg(max_recsize, hit_matrix, limits))

    result = {
        'precision': _my_incremental_mean(precision),
        'recall': _my_incremental_mean(recall),
        'mrr': _my_incremental_mean(mrr),
        'ndcg': _my_incremental_mean(ndcg)
    }

    return result


if __name__ == '__main__':
    print('Building data...')

    batch_size = 1000
    rec_size = 4

    # x = tensor([[0, 1, 0, 0, 1], [0, 0, 0, 0, 0], [1, 1, 1, 0, 1]])
    # y = tensor([[0, 0, 0, 1, 1], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]])
    # x = t_randint(2, size=(100, 10))
    # y = t_randint(2, size=(100, 10))
    x = t_randint(2, size=(100000, 10000))
    y = t_randint(2, size=(100000, 10000))

    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=batch_size)

    print('Building model...')
    tinymodel = _TinyModel()

    print('Evaluation...')
    print(evaluation(tinymodel, loader, rec_size))

    print('Incremental Evaluation...')
    print(incremental_evaluation(tinymodel, loader, rec_size))
