import collections
import os
import random
import types
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import torch
import numpy as np
import json

from evaluation import MetricAccumulator
from util import compute_max_y_aux_popularity, naive_sparse2tensor
from data_loaders import EnsembleDataLoader
from models import EnsembleMultiVAE
from loss_func import ensemble_rvae_rank_pair_loss

# DATASETS ----------------------------------------
CITEULIKE = 'citeulike-a'
EPINIONS = 'epinions'
MOVIELENS_1M = 'ml-1m'
MOVIELENS_20M = 'ml-20m'
NETFLIX = 'netflix'
NETFLIX_SAMPLE = 'netflix_sample'
PINTEREST = 'pinterest'
# ---------------------------------------------------

# SETTINGS ------------------------------------------
dataset_name = MOVIELENS_1M
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# ---------------------------------------------------

SEED = 8734

np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

USE_CUDA = True
CUDA = USE_CUDA and torch.cuda.is_available()

device = torch.device("cuda" if CUDA else "cpu")

print(torch.__version__)

if CUDA:
    print('GPU')
else:
    print('CPU')

data_dir = os.path.expanduser('./data')

data_dir = os.path.join(data_dir, dataset_name)
dataset_file = os.path.join(data_dir, 'data_rvae')

result_dir = os.path.join(data_dir, 'results')
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

run_time = datetime.today().strftime('%Y%m%d_%H%M')

type_str = 'ensemble'
run_dir = os.path.join(result_dir, f'{type_str}_{run_time}')

os.mkdir(run_dir)

# TODO: include learning params
file_model = os.path.join(result_dir, 'best_model.pth')

to_pickle = True

"""## DataLoader 

The dataset is split into train/test with indexes mask

In train the data is reported with a mask of *x* items selected randomly between positive and negative with a proportion 1:2

In test the data is reported with 3 masks of items with less, middle and top popolarity
"""














# Dataloader
trainloader = EnsembleDataLoader(data_dir, use_popularity=True)
n_items = trainloader.n_items
thresholds = trainloader.thresholds
popularity = trainloader.item_popularity
frequencies = trainloader.frequencies

# SETTINGS
settings = types.SimpleNamespace()
settings.dataset_name = os.path.split(data_dir)[-1]
settings.p_dims = [200, 600, n_items]
# settings.p_dims = [1, 5, 10, 50, 100, 200, 600, n_items]
settings.batch_size = 1024
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
max_y_aux_popularity = compute_max_y_aux_popularity(settings)





top_k = (1, 5, 10)


def evaluate(dataloader, tag='validation'):
    # Turn on evaluation mode
    model.eval()
    accumulator = MetricAccumulator()
    result = collections.defaultdict(float)
    batch_num = 0
    n_users_train = 0
    result['train_loss'] = 0
    result['loss'] = 0

    with torch.no_grad():
        for batch_idx, (x, pos, neg, mask, pos_te, neg_te, mask_te, y_a, y_b) in enumerate(
                dataloader.iter_test_ensemble(batch_size=settings.batch_size, tag=tag, device=device)):
            x_tensor = naive_sparse2tensor(x).to(device)
            pos = naive_sparse2tensor(pos).to(device)
            neg = naive_sparse2tensor(neg).to(device)
            mask = naive_sparse2tensor(mask).to(device)
            mask_te = naive_sparse2tensor(mask_te).to(device)

            batch_num += 1
            n_users_train += x_tensor.shape[0]

            x_input = x_tensor * (1 - mask_te)

            y = model(x_input, y_a, y_b, True)

            loss = criterion(x_input, y, pos_items=pos, neg_items=neg, mask=mask)

            result['loss'] += 0  # loss.item()

            recon_batch_cpu = y.cpu().numpy()

            for k in top_k:
                accumulator.compute_metric(x_input.cpu().numpy(),
                                           recon_batch_cpu,
                                           pos_te, neg_te,
                                           popularity,
                                           dataloader.thresholds,
                                           k)

    for k, values in accumulator.get_metrics().items():
        for v in values.metric_names():
            result[f'{v}@{k}'] = values[v]

    return result


# Run
torch.set_printoptions(profile="full")
n_epochs = 1
update_count = 0
settings.batch_size = 1024  # 256
# settings.learning_rate = 1e-4  # 1e-5
settings.learning_rate = 1e-2  # 1e-5
settings.optim = 'adam'
settings.scale = 1000
settings.use_popularity = True
settings.p_dims = [200, 600, n_items]

model = EnsembleMultiVAE(n_items, popularity, thresholds)
model = model.to(device)

criterion = ensemble_rvae_rank_pair_loss(popularity=popularity,
                                scale=settings.scale,
                                thresholds=thresholds,
                                frequencies=frequencies)
best_loss = np.Inf

stat_metric = []

# Test stats
model.eval()
result_test = evaluate(trainloader, 'test')

print('*** TEST RESULTS ***')
print('\n'.join([f'{k:<23}{v}' for k, v in sorted(result_test.items())]))

# Save result

lossname = criterion.__class__.__name__

# result info
with open(os.path.join(run_dir, 'info.txt'), 'w') as fp:
    fp.write(f'K = {settings.gamma_k}\n')
    fp.write(f'Loss = {lossname}\n')
    fp.write(f'Epochs train = {len(stat_metric)}\n')

    fp.write('\n')

    for k in dir(settings):
        if not k.startswith('__'):
            fp.write(f'{k} = {getattr(settings, k)}\n')

# all results
with open(os.path.join(run_dir, 'result.json'), 'w') as fp:
    json.dump(stat_metric, fp)

# test results
with open(os.path.join(run_dir, 'result_test.json'), 'w') as fp:
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

plt.savefig(os.path.join(run_dir, 'loss.png'));

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
        hitRate = [float(x[f'hitrate_by_pop@{k}'].split(',')[j]) for x in stat_metric]

        ax = axes[i]
        i += 1

        ax.plot(hitRate)
        ax.set_title(f'{name} hitrate_by_pop@{k}')

plt.savefig(os.path.join(run_dir, 'hr.png'));

print('DONE', run_dir)
