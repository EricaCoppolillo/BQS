import collections
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import torch
import numpy as np
import json

from evaluation import MetricAccumulator
from util import compute_max_y_aux_popularity, naive_sparse2tensor, set_seed
from data_loaders import EnsembleDataLoader
from models import EnsembleMultiVAE
from loss_func import ensemble_rvae_rank_pair_loss
from config import Config

# SETTINGS ------------------------------------------
datasets = Config("./datasets_info.json")
dataset_name = datasets.MOVIELENS_1M
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
config = Config("./rvae_config.json")
config.metrics_scale = eval(config.metrics_scale)
config.use_popularity = eval(config.use_popularity)
config.p_dims = eval(config.p_dims)
# ---------------------------------------------------

SEED = 8734
set_seed(SEED)

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
config.p_dims.append(n_items)
thresholds = trainloader.thresholds
popularity = trainloader.item_popularity
frequencies = trainloader.frequencies

max_y_aux_popularity = compute_max_y_aux_popularity(config)
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
                dataloader.iter_test_ensemble(batch_size=config.batch_size, tag=tag, device=device)):
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

model = EnsembleMultiVAE(n_items, popularity, thresholds)
model = model.to(device)

criterion = ensemble_rvae_rank_pair_loss(popularity=popularity,
                                scale=config.scale,
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
    fp.write(f'K = {config.gamma_k}\n')
    fp.write(f'Loss = {lossname}\n')
    fp.write(f'Epochs train = {len(stat_metric)}\n')

    fp.write('\n')

    for k in config.__dict__:
        fp.write(f'{k} = {config.__dict__[k]}\n')

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
