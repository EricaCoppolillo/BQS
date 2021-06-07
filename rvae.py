import collections
import os
import random
import types
from datetime import datetime
import matplotlib.pyplot as plt
import time
from evaluation import MetricAccumulator
from util import *
from models import MultiVAE
from loss_func import rvae_rank_pair_loss
from data_loaders import DataLoader

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
BASELINE = False  # Baseline/LowPop model
dataset_name = MOVIELENS_1M
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# ---------------------------------------------------

SEED = 8734

np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

USE_CUDA = True
CUDA = USE_CUDA and torch.cuda.is_available()

device = torch.device("cuda" if CUDA else "cpu")

print(torch.__version__)

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
if BASELINE:
    type_str = 'baseline'
else:
    type_str = 'popularity_low'
run_dir = os.path.join(result_dir, f'{type_str}_{run_time}')

os.mkdir(run_dir)

# TODO: include learning params
file_model = os.path.join(run_dir, 'best_model.pth')

to_pickle = True

"""# Train and test"""

if BASELINE:
    trainloader = DataLoader(dataset_file, use_popularity=False)
else:
    trainloader = DataLoader(dataset_file, use_popularity=True)

n_items = trainloader.n_items

# SETTINGS
settings = types.SimpleNamespace()
settings.dataset_name = os.path.split(data_dir)[-1]
# settings.p_dims = [200, 600, n_items]

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

popularity = trainloader.item_popularity
thresholds = trainloader.thresholds
frequencies = trainloader.frequencies

max_y_aux_popularity = compute_max_y_aux_popularity()



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
        # print(f'batches are {dataloader.n_items // settings.batch_size} with size {settings.batch_size}')

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
    accumulator = MetricAccumulator()
    result = collections.defaultdict(float)
    batch_num = 0
    n_users_train = 0
    # n_positives_predicted = np.zeros(3)
    result['train_loss'] = 0
    result['loss'] = 0

    with torch.no_grad():
        for batch_idx, (x, pos, neg, mask, pos_te, neg_te, mask_te) in enumerate(
                dataloader.iter_test(batch_size=settings.batch_size, tag=tag)):
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

n_epochs = 50
update_count = 0
settings.optim = 'adam'
settings.scale = 1000
settings.use_popularity = True
settings.p_dims = [200, 600, n_items]

print(settings.p_dims)
model = MultiVAE(settings.p_dims)
model = model.to(device)

criterion = rvae_rank_pair_loss(popularity=popularity if settings.use_popularity else None,
                                scale=settings.scale,
                                thresholds=thresholds,
                                frequencies=frequencies)
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
        optimizer = torch.optim.SGD(params=model.parameters(), lr=settings.learning_rate, momentum=0.9,
                                    dampening=0, weight_decay=0, nesterov=True)
    else:
        optimizer = torch.optim.RMSprop(params=model.parameters(), lr=settings.learning_rate, alpha=0.99, eps=1e-08,
                                        weight_decay=settings.weight_decay, momentum=0, centered=False)

    for epoch in range(1, n_epochs + 1):
        epoch_start_time = time.time()
        train_loss = train(trainloader, epoch, optimizer)
        result = evaluate(trainloader, popularity)

        result['train_loss'] = train_loss
        stat_metric.append(result)

        print_metric = lambda k, v: f'{k}: {v:.4f}' if not isinstance(v, str) else f'{k}: {v}'
        ss = ' | '.join([print_metric(k, v) for k, v in stat_metric[-1].items() if k in
                         ('train_loss', 'loss', 'hitrate@5', 'hitrate_by_pop@5', 'weighted_luciano_stat@5',
                          'luciano_stat_by_pop@5')])
        ss = f'| Epoch {epoch:3d} | time: {time.time() - epoch_start_time:4.2f}s | {ss} |'
        ls = len(ss)
        print('-' * ls)
        print(ss)
        print('-' * ls)

        # Save the model if the n100 is the best we've seen so far.
        if best_loss > result['loss']:
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
ax3.set_title('HitRate@5')

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

plt.show();

"""# Test stats"""

model.eval()
result_test = evaluate(trainloader, popularity, 'test')

print(f'K = {settings.gamma_k}')
print('\n'.join([f'{k:<23}{v}' for k, v in sorted(result_test.items())]))

"""# Save result"""

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

#    fp.write('\n' * 4)
#    model_print, _ = torchsummary.summary_string(model, (dataloader.n_items,), device='gpu' if CUDA else 'cpu')
#    fp.write(model_print)


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
