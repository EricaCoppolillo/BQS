import collections
import os
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import time
import torch
import numpy as np
import json
from evaluation import MetricAccumulator
from util import compute_max_y_aux_popularity, naive_sparse2tensor, set_seed
from models import MultiVAE
from loss_func import rvae_rank_pair_loss
from data_loaders import DataLoader
from config import Config

datasets = Config("./datasets_info.json")
model_types = Config("./model_type_info.json")

# SETTINGS ------------------------------------------
config = Config("./rvae_config.json")

os.environ['CUDA_VISIBLE_DEVICES'] = config.CUDA_VISIBLE_DEVICES
dataset_name = eval(config.dataset_name)
model_type = eval(config.model_type)
copy_pasting_data = eval(config.copy_pasting_data)
config.metrics_scale = eval(config.metrics_scale)
config.use_popularity = eval(config.use_popularity)
config.p_dims = eval(config.p_dims)
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
dataset_file = os.path.join(data_dir, 'data_rvae')

result_dir = os.path.join(data_dir, 'results')
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

run_time = datetime.today().strftime('%Y%m%d_%H%M')
run_dir = os.path.join(result_dir, f'{model_type}_{run_time}')

os.mkdir(run_dir)

# TODO: include learning params
file_model = os.path.join(run_dir, 'best_model.pth')

to_pickle = True

""" Train and test"""

if model_type == model_types.BASELINE:
    trainloader = DataLoader(dataset_file, seed=SEED, decreasing_factor=1, use_popularity=False)
else:
    trainloader = DataLoader(dataset_file, seed=SEED, decreasing_factor=config.decreasing_factor, use_popularity=True)

n_items = trainloader.n_items
config.p_dims.append(n_items)
popularity = trainloader.item_popularity
thresholds = trainloader.thresholds
frequencies = trainloader.frequencies

max_y_aux_popularity = compute_max_y_aux_popularity(config)


def train(dataloader, epoch, optimizer):
    global update_count

    log_interval = int(trainloader.n_users * .7 / config.batch_size // 4)
    if log_interval == 0:
        log_interval = 1

    # Turn on training mode
    model.train()
    train_loss = 0.0
    train_loss_cumulative = 0.0
    start_time = time.time()

    if epoch == 1:
        print(f'log every {log_interval} log interval')
        # print(f'batches are {dataloader.n_items // settings.batch_size} with size {settings.batch_size}')

    for batch_idx, (x, pos, neg, mask) in enumerate(dataloader.iter(batch_size=config.batch_size)):
        x = naive_sparse2tensor(x).to(device)
        pos_items = naive_sparse2tensor(pos).to(device)
        neg_items = naive_sparse2tensor(neg).to(device)
        mask = naive_sparse2tensor(mask).to(device)

        update_count += 1
        if config.total_anneal_steps > 0:
            anneal = min(config.anneal_cap, 1. * update_count / config.total_anneal_steps)
        else:
            anneal = config.anneal_cap

        # TRAIN on batch
        optimizer.zero_grad()
        y, mu, logvar = model(x)

        # loss = criterion(recon_batch, x, pos_items, neg_items, mask, mask, mu, logvar, anneal)
        loss = criterion(x, y, mu, logvar, anneal, pos_items=pos_items, neg_items=neg_items, mask=mask,
                         model_type=model_type)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_loss_cumulative += loss.item()

        update_count += 1

        if batch_idx % log_interval == 0 and batch_idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:4d}/{:4d} batches | ms/batch {:4.2f} | loss {:4.4f}'.format(
                epoch, batch_idx, len(range(0, dataloader.size['train'], config.batch_size)),
                elapsed * 1000 / log_interval,
                train_loss / log_interval))

            start_time = time.time()
            train_loss = 0.0

        if CUDA:
            torch.cuda.empty_cache()
            # print(f"[TRAIN]: Cuda cache emptied")

    return train_loss_cumulative / (1 + batch_idx)


top_k = (1, 5, 10)


def evaluate(dataloader, normalized_popularity, tag='validation'):
    # print("STARTING TO EVAL")
    # Turn on evaluation mode
    model.eval()
    accumulator = MetricAccumulator()
    result = collections.defaultdict(float)
    batch_num = 0
    n_users_train = 0
    # n_positives_predicted = np.zeros(3)
    result['train_loss'] = 0
    result['loss'] = 0

    with torch.no_grad():
        for batch_idx, (x, pos, neg, mask, pos_te, neg_te, mask_te) in enumerate(
                dataloader.iter_test(batch_size=config.batch_size, tag=tag)):
            # print(f"[EVAL]: Entering batch {batch_idx}")
            x_tensor = naive_sparse2tensor(x).to(device)
            pos = naive_sparse2tensor(pos).to(device)
            neg = naive_sparse2tensor(neg).to(device)
            mask = naive_sparse2tensor(mask).to(device)
            mask_te = naive_sparse2tensor(mask_te).to(device)
            # print("[EVAL]: batch data have been processed")
            batch_num += 1
            n_users_train += x_tensor.shape[0]

            x_input = x_tensor * (1 - mask_te)
            y, mu, logvar = model(x_input, True)
            # print("[EVAL]: model forward done")
            #            loss = criterion(recon_batch, x_input, pos, neg, mask, mask, mu, logvar)
            loss = criterion(x_input, y, mu, logvar, 0, pos_items=pos, neg_items=neg, mask=mask, model_type=model_type)
            # print("[EVAL]: Loss computed")
            result['loss'] += loss.item()

            recon_batch_cpu = y.cpu().numpy()

            for k in top_k:
                # print(f"[EVAL]: Computing metric for k={k}")
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


torch.set_printoptions(profile="full")

n_epochs = config.n_epochs
update_count = 0

print(config.p_dims)
model = MultiVAE(config.p_dims)
model = model.to(device)

criterion = rvae_rank_pair_loss(device=device, popularity=popularity if config.use_popularity else None,
                                scale=config.scale,
                                thresholds=thresholds,
                                frequencies=frequencies)
# criterion = rvae_focal_loss(popularity=popularity if settings.use_popularity else None, scale=settings.scale)

best_loss = -np.Inf

stat_metric = []

print('At any point you can hit Ctrl + C to break out of training early.')
try:
    if config.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=config.learning_rate,
                                     weight_decay=config.weight_decay)
    elif config.optim == 'sgd':
        optimizer = torch.optim.SGD(params=model.parameters(), lr=config.learning_rate, momentum=0.9,
                                    dampening=0, weight_decay=0, nesterov=True)
    else:
        optimizer = torch.optim.RMSprop(params=model.parameters(), lr=config.learning_rate, alpha=0.99, eps=1e-08,
                                        weight_decay=config.weight_decay, momentum=0, centered=False)

    for epoch in range(1, n_epochs + 1):
        epoch_start_time = time.time()
        train_loss = train(trainloader, epoch, optimizer)
        result = evaluate(trainloader, popularity)

        result['train_loss'] = train_loss
        stat_metric.append(result)
        renaming_luciano_stat = {"weighted_luciano_stat@5": "weighted_hit_rate@5",
                                 "luciano_stat_by_pop@5": "hitrate_by_pop@5",
                                 "train_loss": "train_loss", "loss": "loss"}
        print_metric = lambda k, v: f'{renaming_luciano_stat[k]}: {v:.4f}' if not isinstance(v, str) \
            else f'{renaming_luciano_stat[k]}: {v}'
        ss = ' | '.join([print_metric(k, v) for k, v in stat_metric[-1].items() if k in
                         ('train_loss', 'loss', 'weighted_luciano_stat@5',
                          'luciano_stat_by_pop@5')])
        ss = f'| Epoch {epoch:3d} | time: {time.time() - epoch_start_time:4.2f}s | {ss} |'
        ls = len(ss)
        print('-' * ls)
        print(ss)
        print('-' * ls)

        # Save the model if the n100 is the best we've seen so far.
        '''
        if best_loss > result['loss']:
            torch.save(model.state_dict(), file_model)
            best_loss = result['loss']
        '''
        # TODO: automating this code block with respect to the model type (e.g.: baseline, low, etc.)
        LOW, MED, HIGH = 0, 1, 2
        val_result = float(result["luciano_stat_by_pop@5"].split(",")[LOW])
        if best_loss < val_result:
            torch.save(model.state_dict(), file_model)
            best_loss = val_result

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# output.show()

model = MultiVAE(config.p_dims)
model = model.to(device)
model.load_state_dict(torch.load(file_model, map_location=device))
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
result_test = evaluate(trainloader, popularity, 'test')

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
    main_directory = os.path.join('./data', dataset_name, model_type)
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
