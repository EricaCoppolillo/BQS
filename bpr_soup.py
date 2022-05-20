import collections
import os
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import time
import torch
import numpy as np
import json
import pathlib
from torch import nn

import tqdm

from evaluation import MetricAccumulator
from util import compute_max_y_aux_popularity, naive_sparse2tensor, set_seed
from models import BPR
from loss_func import bpr_loss
from data_loaders import BprDataLoader, CachedBprDataLoader, BprJannachDataLoader, BprBorattoNegativeSamplingDataLoader
from config import Config

datasets = Config("./datasets_info.json")
model_types = Config("./model_type_info.json")

# SETTINGS ------------------------------------------
path_to_config = sys.argv[1]  # "./bpr_config.json"
config = Config(path_to_config)

if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = config.CUDA_VISIBLE_DEVICES

dataset_name = eval(config.dataset_name)
model_type = eval(config.model_type)
copy_pasting_data = eval(config.copy_pasting_data)
config.cached_dataloader = eval(config.cached_dataloader if 'cached_dataloader' in config else 'False')
config.metrics_scale = eval(config.metrics_scale)
config.latent_dim = eval(config.latent_dim)
config.alpha = eval(config.alpha)
config.gamma = eval(config.gamma)
reg_weight = float(config.reg_weight)
pc_weight = float(config.pc_weight)
if config.gamma:
    config.gamma = float(config.gamma)
if config.alpha:
    config.alpha = float(config.alpha)
WIDTH_PARAM = float(config.jannach_width)
if model_type == "reweighting":
    assert (not config.alpha or not config.gamma) or (config.alpha > 0 and config.gamma > 0), config.alpha
# ---------------------------------------------------

# set seed for experiment reproducibility
SEED = config.seed
set_seed(SEED)

USE_CUDA = True
CUDA = USE_CUDA and torch.cuda.is_available()
device = torch.device("cuda" if CUDA else "cpu")

print(f"Current PyTorch version: {torch.__version__}")

if CUDA:
    print('run on cuda %s' % os.environ['CUDA_VISIBLE_DEVICES'])
else:
    print('cuda not available')

data_dir = os.path.expanduser('./data')

data_dir = os.path.join(data_dir, dataset_name)
dataset_file = os.path.join(data_dir, 'data_bpr')

result_dir = os.path.join(data_dir, 'bpr', 'results')
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

run_time = datetime.today().strftime('%Y%m%d_%H%M%S')
run_dir = os.path.join(result_dir, f'{model_type}{config.data_loader_type}_{run_time}')

os.makedirs(run_dir, exist_ok=True)

# TODO: include learning params
file_model = os.path.join(run_dir, 'best_model.pth')

to_pickle = True

""" Train and test"""
if not config.cached_dataloader:
    if model_type == model_types.BASELINE:
        if config.data_loader_type == "":
            trainloader = BprDataLoader(dataset_file, seed=SEED, decreasing_factor=1, model_type=model_type)
    else:
        trainloader = BprDataLoader(dataset_file, seed=SEED, decreasing_factor=config.decreasing_factor,
                                    model_type=model_type, alpha=config.alpha, gamma=config.gamma)
else:
    print('USE CACHED DATALOADER')
    if model_type == model_types.BASELINE:
        trainloader = CachedBprDataLoader(dataset_file, seed=SEED, decreasing_factor=1, model_type=model_type)
    else:
        trainloader = CachedBprDataLoader(dataset_file, seed=SEED, decreasing_factor=config.decreasing_factor,
                                          model_type=model_type, alpha=config.alpha, gamma=config.gamma)

n_items = trainloader.n_items
n_users = trainloader.n_users
popularity = trainloader.item_popularity_dict["training"]
thresholds = trainloader.thresholds
abs_thresholds = trainloader.absolute_thresholds
abs_frequencies = trainloader.absolute_item_popularity_dict['training']
frequencies = trainloader.frequencies_dict["training"]

print(f"\nRegularizer: {config.regularizer}")

if config.regularizer == "JS":
    def _determine_class_pop(freq, th):
        if freq <= th[0]:
            return 0
        elif freq > th[1]:
            return 2
        else:
            return 1


    pop_mapping = torch.zeros(size=(n_items, 3))
    pop_mapping = pop_mapping.to(device)
    for i in range(len(abs_frequencies)):
        pop_mapping[i, _determine_class_pop(abs_frequencies[i], abs_thresholds)] = 1
    pop_mapping.requires_grad = False
elif config.regularizer == "boratto":
    if CUDA:
        torch_pop = torch.tensor(popularity).float().cuda()
    else:
        torch_pop = torch.tensor(popularity).float()
    torch_pop.requires_grad = False
elif config.regularizer == "PD":
    elu_func = torch.nn.ELU()
    if CUDA:
        torch_pop = torch.tensor([elem ** float(config.reg_weight) for elem in popularity]).float().cuda()
    else:
        torch_pop = torch.tensor([elem ** float(config.reg_weight) for elem in popularity]).float()
    torch_pop.requires_grad = False
elif config.regularizer == "PC":
    if CUDA:
        torch_pop = torch.tensor(popularity).float().cuda()
        torch_abs_pop = torch.tensor(abs_frequencies).float().cuda()
        torch_n_items = torch.tensor([n_items] * config.batch_size).float().cuda()
    else:
        torch_pop = torch.tensor(popularity).float()
        torch_abs_pop = torch.tensor(abs_frequencies).float()
        torch_n_items = torch.tensor([n_items] * config.batch_size).float()
    torch_pop.requires_grad = False
    torch_abs_pop.requires_grad = False
    torch_n_items.requires_grad = False

top_k = (1, 5, 10)

def evaluate(dataloader, tag='validation'):
    # print("STARTING TO EVAL")
    # Turn on evaluation mode
    model.eval()
    accumulator = MetricAccumulator()
    result = collections.defaultdict(float)
    batch_num = 0
    n_users_train = 0
    train_loss_cumulative = 0
    result['train_loss'] = 0

    # compute loss

    if model_type != model_types.LOW:
        for triplets, mask in tqdm.tqdm(dataloader.iter(batch_size=config.batch_size, tag=tag),
                                        desc=f'Compute {tag} loss'):
            users = torch.tensor(triplets[:, 0], device=device)
            pos = torch.tensor(triplets[:, 1], device=device)
            neg = torch.tensor(triplets[:, 2], device=device)
            mask = torch.tensor(mask, device=device)

            y = model(users, pos, neg)

            if config.regularizer == "PD":
                y = elu_func(y) + 1

            loss = criterion(y, mask)

            if config.regularizer == "boratto":
                # computing the absolute correlation
                vx = loss - torch.mean(loss)
                vy = torch_pop[pos.long()] - torch.mean(torch_pop[pos.long()])
                correlation = torch.abs(
                    torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2))))
                loss = torch.sum(loss)
                loss += correlation * reg_weight
            else:
                loss = torch.sum(loss)

            train_loss_cumulative += loss.item()
            batch_num += 1

        if CUDA:
            torch.cuda.empty_cache()

        result['loss'] = train_loss_cumulative / batch_num if batch_num > 0 else 0
    else:
        result['loss'] = 0.

    bs = 256
    if n_items > 20000:
        bs = 16

    # evaluate
    all_items = torch.arange(0, n_items, device=device, dtype=torch.long)
    all_items = all_items.repeat(bs, 1)

    with torch.no_grad():
        for batch_idx, (user_id, positives, negatives) in enumerate(dataloader.iter_test(batch_size=bs, tag=tag)):
            # print(f"[EVAL]: Entering batch {batch_idx}")
            user = torch.tensor(user_id, device=device)
            # pos = torch.tensor(positives, device=device)
            # neg = torch.tensor(negatives, device=device)

            # print("[EVAL]: batch data have been processed")
            # all_items = torch.cat([pos, neg], -1)
            if user.shape[0] != all_items.shape[0]:
                all_items = all_items[0:user.shape[0]]
            score = model.score(user, all_items)

            n_users_train += 1

            recon_batch_cpu = score.cpu().numpy()

            for k in top_k:
                # print(f"[EVAL]: Computing metric for k={k}")
                accumulator.compute_metric(None,
                                           recon_batch_cpu,
                                           positives, negatives,
                                           popularity,
                                           dataloader.thresholds,
                                           k)

    for k, values in accumulator.get_metrics().items():
        for v in values.metric_names():
            result[f'{v}@{k}'] = values[v]

    return result


torch.set_printoptions(profile="full")

n_epochs = 0
update_count = 0

curr_dir = pathlib.Path.cwd()
model = BPR(n_users=n_users, n_items=n_items, n_factors=config.latent_dim)
model = model.to(device)
if len(sys.argv) == 5:
    print('path are provided as input')
    baseline_file_model = curr_dir / 'data' / dataset_name / 'bpr' / 'results' / sys.argv[2] / 'best_model.pth'
    oversampling_file_model = curr_dir / 'data' / dataset_name / 'bpr' / 'results' / sys.argv[3] / 'best_model.pth'
    low_file_model = curr_dir / 'data' / dataset_name / 'bpr' / 'results' / sys.argv[4] / 'best_model.pth'
else:
    baseline_file_model = curr_dir / 'data' / dataset_name / 'baseline' / 'best_model.pth'
    oversampling_file_model = curr_dir / 'data' / dataset_name / 'oversampling' / 'best_model.pth'
    low_file_model = curr_dir / 'data' / dataset_name / 'low' / 'best_model.pth'
model.load_state_dict(torch.load(baseline_file_model))

oversampling_model = BPR(n_users=n_users, n_items=n_items, n_factors=config.latent_dim)
oversampling_model = oversampling_model.to(device)
oversampling_model.load_state_dict(torch.load(oversampling_file_model))

low_model = BPR(n_users=n_users, n_items=n_items, n_factors=config.latent_dim)
low_model = low_model.to(device)
low_model.load_state_dict(torch.load(low_file_model))

alpha = float(eval(config.soup_weight))
beta = (1. - alpha) / 2
model.embed_user.weight = nn.Parameter(
    (beta*model.embed_user.weight + beta*oversampling_model.embed_user.weight + alpha * low_model.embed_user.weight))
model.embed_item.weight = nn.Parameter(
    (beta*model.embed_item.weight + beta*oversampling_model.embed_item.weight + alpha * low_model.embed_item.weight))

criterion = bpr_loss(is_weighted_model=model_type == model_types.REWEIGHTING)

best_loss = np.Inf

stat_metric = []

print('At any point you can hit Ctrl + C to break out of training early.')
try:
    result = evaluate(trainloader)

    result['train_loss'] = 0.0
    stat_metric.append(result)
    renaming_luciano_stat = {"weighted_luciano_stat@5": "weighted_hit_rate@5",
                             "luciano_stat_by_pop@5": "hitrate_by_pop@5",
                             "train_loss": "train_loss", "loss": "loss"}
    print_metric = lambda k, v: f'{renaming_luciano_stat[k]}: {v:.4f}' if not isinstance(v, str) \
        else f'{renaming_luciano_stat[k]}: {v}'
    ss = ' | '.join([print_metric(k, v) for k, v in stat_metric[-1].items() if k in
                     ('train_loss', 'loss', 'weighted_luciano_stat@5',
                      'luciano_stat_by_pop@5')])
    ls = len(ss)
    print('-' * ls)
    print(ss)
    print('-' * ls)
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

model.eval()

"""# Training stats"""

renaming_luciano_stat = {"loss": "loss", "train_loss": "train_loss"}
d1 = {f"luciano_recalled_by_pop@{k}": f"recall_by_pop@{k}" for k in [1, 5, 10]}
d2 = {f"luciano_stat_by_pop@{k}": f"hit_rate_by_pop@{k}" for k in [1, 5, 10]}
d3 = {f"luciano_stat@{k}": f"hit_rate@{k}" for k in [1, 5, 10]}
d4 = {f"luciano_weighted_stat@{k}": f"weighted_hit_Rate@{k}" for k in [1, 5, 10]}
renaming_luciano_stat = {**renaming_luciano_stat, **d1, **d2, **d3, **d4}

print("Training Statistics: \n")
print('\n'.join([f'{renaming_luciano_stat[k]:<23}{v}' for k, v in sorted(stat_metric[-1].items())
                 if k in renaming_luciano_stat]))

# LOSS
lossTrain = [x['train_loss'] for x in stat_metric]
lossTest = [x['loss'] for x in stat_metric]
lastHitRate = [x['luciano_stat@5'] for x in stat_metric]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 8))
ax1.plot(lossTrain, color='b', )
# ax1.set_yscale('log')
ax1.set_title('Train')

ax2.plot(lossTest, color='r')
# ax2.set_yscale('log')
ax2.set_title('Validation')

ax3.plot(lastHitRate)
ax3.set_title('HitRate@5')

fig, axes = plt.subplots(4, 3, figsize=(20, 20))

axes = axes.ravel()
i = 0

for k in top_k:
    hitRate = [x[f'luciano_stat@{k}'] for x in stat_metric]

    ax = axes[i]
    i += 1

    ax.plot(hitRate)
    ax.set_title(f'HitRate@{k}')

for j, name in enumerate('LessPop MiddlePop TopPop'.split()):
    for k in top_k:
        hitRate = [float(x[f'luciano_stat_by_pop@{k}'].split(',')[j]) for x in stat_metric]

        ax = axes[i]
        i += 1

        ax.plot(hitRate)
        ax.set_title(f'{name} hitrate_by_pop@{k}')

plt.show();

"""# Test stats"""

model.eval()
result_test = evaluate(trainloader, 'test')
result_validation = evaluate(trainloader, 'validation')

print(f'K = {config.gamma_k}')
print("Test Statistics: \n")
print('\n'.join([f'{renaming_luciano_stat[k]:<23}{v}' for k, v in sorted(result_test.items())
                 if k in renaming_luciano_stat]))

"""# Save result"""

lossname = criterion.__class__.__name__

# result info
with open(os.path.join(run_dir, 'info.txt'), 'w') as fp:
    fp.write(f'K = {config.gamma_k}\n')
    fp.write(f'Loss = {lossname}\n')
    # fp.write(f'Epochs train = {len(stat_metric)}\n')
    fp.write(f"Dataset = {dataset_name}\n")
    fp.write(f"Seed = {SEED}\n")
    fp.write('\n')

    for k in config.__dict__:
        fp.write(f'{k} = {config.__dict__[k]}\n')


#    fp.write('\n' * 4)
#    model_print, _ = torchsummary.summary_string(model, (dataloader.n_items,), device='gpu' if CUDA else 'cpu')
#    fp.write(model_print)


# all results
def renaming_results(result_dict, rename_dict):
    return {rename_dict[k]: v for k, v in result_dict.items() if k in rename_dict}


with open(os.path.join(run_dir, 'result.json'), 'w') as fp:
    json.dump(list(map(lambda x: renaming_results(x, renaming_luciano_stat), stat_metric)), fp)

# validation results
with open(os.path.join(run_dir, 'result_val.json'), 'w') as fp:
    json.dump(renaming_results(result_validation, renaming_luciano_stat), fp,
              indent=4, sort_keys=True)

# test results
with open(os.path.join(run_dir, 'result_test.json'), 'w') as fp:
    json.dump(renaming_results(result_test, renaming_luciano_stat), fp,
              indent=4, sort_keys=True)

# chart 1
lossTrain = [x['train_loss'] for x in stat_metric]
lossTest = [x['loss'] for x in stat_metric]

lastHitRate = [x['luciano_stat@5'] for x in stat_metric]

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
plt.savefig(os.path.join(run_dir, 'loss.pdf'));

# chart 2
fig, axes = plt.subplots(4, 3, figsize=(20, 20))

axes = axes.ravel()
i = 0

for k in top_k:
    hitRate = [x[f'luciano_stat@{k}'] for x in stat_metric]

    ax = axes[i]
    i += 1

    ax.plot(hitRate)
    ax.set_title(f'HitRate@{k}')

for j, name in enumerate('LessPop MiddlePop TopPop'.split()):
    for k in top_k:
        hitRate = [float(x[f'luciano_stat_by_pop@{k}'].split(',')[j]) for x in stat_metric]

        ax = axes[i]
        i += 1

        ax.plot(hitRate)
        ax.set_title(f'{name} hitrate_by_pop@{k}')

plt.savefig(os.path.join(run_dir, 'hr.png'));
plt.savefig(os.path.join(run_dir, 'hr.pdf'));

if copy_pasting_data:
    main_directory = os.path.join('./data', dataset_name, "bpr", model_type)
    # deleting the main directory used by the ensemble script
    import shutil

    if os.path.isdir(main_directory):
        shutil.rmtree(main_directory, ignore_errors=True)
    # moving the run directory into the main
    from distutils.dir_util import copy_tree

    os.makedirs(main_directory)
    copy_tree(run_dir, main_directory)

print('DONE')
print(run_dir)
