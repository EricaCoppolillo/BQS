import collections
import os
from datetime import datetime
import matplotlib.pyplot as plt
import time
import torch
import numpy as np
import json
from evaluation import MetricAccumulator
from util import naive_sparse2tensor, set_seed, normalize_distr, js_div_2d
from models import MultiVAE
from loss_func import rvae_rank_pair_loss
import data_loaders
import stream_data_loaders
from config import Config

datasets = Config("./datasets_info.json")
model_types = Config("./model_type_info.json")

# SETTINGS ------------------------------------------
config = Config("./rvae_config.json")

if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = config.CUDA_VISIBLE_DEVICES

dataset_name = eval(config.dataset_name)
model_type = eval(config.model_type)
copy_pasting_data = eval(config.copy_pasting_data)
config.cached_dataloader = eval(config.cached_dataloader if 'cached_dataloader' in config else 'False')
config.metrics_scale = eval(config.metrics_scale)
config.use_popularity = eval(config.use_popularity)
config.p_dims = eval(config.p_dims)
config.alpha = eval(config.alpha)
config.gamma = eval(config.gamma)
reg_weight = float(config.reg_weight)
pc_weight = float(config.pc_weight)
if config.gamma:
    config.gamma = float(config.gamma)
if config.alpha:
    config.alpha = float(config.alpha)
WIDTH_PARAM = float(config.jannach_width)
BETA_SAMPLING = float(config.beta_sampling)
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
dataset_file = os.path.join(data_dir, 'data_rvae')

result_dir = os.path.join(data_dir, 'results')
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

run_time = datetime.today().strftime('%Y%m%d_%H%M')
run_dir = os.path.join(result_dir, f'{model_type}{config.data_loader_type}_{run_time}')

os.makedirs(run_dir, exist_ok=True)

# TODO: include learning params
file_model = os.path.join(run_dir, 'best_model.pth')

to_pickle = True

""" Train and test"""

if not config.cached_dataloader:
    print(f"Model: {model_type}\nData Loader: {config.data_loader_type}")
    if model_type == model_types.BASELINE:
        if config.data_loader_type == "inverse_ppr":
            trainloader = data_loaders.InversePersonalizedPagerankNegativeSamplingDataLoader(dataset_file, seed=SEED,
                                                                                             decreasing_factor=1,
                                                                                             model_type=model_type)
        elif config.data_loader_type == "ppr":
            trainloader = data_loaders.PersonalizedPagerankNegativeSamplingDataLoader(dataset_file, seed=SEED,
                                                                                      decreasing_factor=1,
                                                                                      model_type=model_type)
        elif config.data_loader_type == "jannach":
            trainloader = data_loaders.JannachDataLoader(file_tr=dataset_file, seed=SEED, decreasing_factor=1,
                                                         model_type=model_type,
                                                         width_param=WIDTH_PARAM)
        elif config.data_loader_type == "boratto":
            trainloader = data_loaders.BorattoNegativeSamplingDataLoader(file_tr=dataset_file, seed=SEED,
                                                                         decreasing_factor=1,
                                                                         model_type=model_type)
        elif config.data_loader_type == "negative":
            trainloader = data_loaders.NegativeSamplingDataLoader(dataset_file, seed=SEED, decreasing_factor=1,
                                                                  model_type=model_type)
        elif config.data_loader_type == "word2vec":
            trainloader = data_loaders.Word2VecNegativeSamplingDataLoader(file_tr=dataset_file, seed=SEED,
                                                                          decreasing_factor=1,
                                                                          model_type=model_type,
                                                                          beta_sampling=BETA_SAMPLING)
        elif config.data_loader_type == "stream":
            trainloader = stream_data_loaders.StreamDataLoader(dataset_file, seed=SEED, decreasing_factor=1,
                                                               model_type=model_type)
        elif config.data_loader_type == "2stages":
            trainloader = stream_data_loaders.TwoStagesNegativeSamplingDataLoader(file_tr=dataset_file, seed=SEED,
                                                                                  decreasing_factor=1,
                                                                                  model_type=model_type,
                                                                                  beta_sampling=BETA_SAMPLING,
                                                                                  device=device,
                                                                                  model=None, model_class="rvae")
        else:
            trainloader = data_loaders.DataLoader(dataset_file, seed=SEED, decreasing_factor=1, model_type=model_type)
    else:
        if config.data_loader_type == "inverse_ppr":
            trainloader = data_loaders.InversePersonalizedPagerankNegativeSamplingDataLoader(dataset_file,
                                                                                             seed=SEED,
                                                                                             decreasing_factor=config.decreasing_factor,
                                                                                             model_type=model_type,
                                                                                             alpha=config.alpha,
                                                                                             gamma=config.gamma)
        elif config.data_loader_type == "ppr":
            trainloader = data_loaders.PersonalizedPagerankNegativeSamplingDataLoader(dataset_file,
                                                                                      seed=SEED,
                                                                                      decreasing_factor=config.decreasing_factor,
                                                                                      model_type=model_type,
                                                                                      alpha=config.alpha,
                                                                                      gamma=config.gamma)
        elif config.data_loader_type == "jannach":
            trainloader = data_loaders.JannachDataLoader(dataset_file, seed=SEED,
                                                         decreasing_factor=config.decreasing_factor,
                                                         model_type=model_type, alpha=config.alpha, gamma=config.gamma,
                                                         width_param=WIDTH_PARAM)
        elif config.data_loader_type == "boratto":
            trainloader = data_loaders.BorattoNegativeSamplingDataLoader(dataset_file, seed=SEED,
                                                                         decreasing_factor=config.decreasing_factor,
                                                                         model_type=model_type, alpha=config.alpha,
                                                                         gamma=config.gamma)
        elif config.data_loader_type == "negative":
            trainloader = data_loaders.NegativeSamplingDataLoader(dataset_file, seed=SEED,
                                                                  decreasing_factor=config.decreasing_factor,
                                                                  model_type=model_type, alpha=config.alpha,
                                                                  gamma=config.gamma)
        elif config.data_loader_type == "word2vec":
            trainloader = data_loaders.Word2VecNegativeSamplingDataLoader(dataset_file, seed=SEED,
                                                                          decreasing_factor=config.decreasing_factor,
                                                                          model_type=model_type, alpha=config.alpha,
                                                                          gamma=config.gamma,
                                                                          beta_sampling=BETA_SAMPLING)
        elif config.data_loader_type == "stream":
            trainloader = stream_data_loaders.StreamDataLoader(dataset_file, seed=SEED,
                                                               decreasing_factor=config.decreasing_factor,
                                                               model_type=model_type, alpha=config.alpha,
                                                               gamma=config.gamma)
        elif config.data_loader_type == "2stages":
            trainloader = stream_data_loaders.TwoStagesNegativeSamplingDataLoader(file_tr=dataset_file, seed=SEED,
                                                                                  decreasing_factor=config.decreasing_factor,
                                                                                  model_type=model_type,
                                                                                  alpha=config.alpha,
                                                                                  gamma=config.gamma,
                                                                                  beta_sampling=BETA_SAMPLING,
                                                                                  device=device,
                                                                                  model=None, model_class="rvae")
        else:
            trainloader = data_loaders.DataLoader(dataset_file, seed=SEED, decreasing_factor=config.decreasing_factor,
                                                  model_type=model_type, alpha=config.alpha, gamma=config.gamma)
else:
    print('USE CACHED DATALOADER')
    if model_type == model_types.BASELINE:
        trainloader = data_loaders.CachedDataLoader(dataset_file, seed=SEED, decreasing_factor=1, model_type=model_type)
    else:
        trainloader = data_loaders.CachedDataLoader(dataset_file, seed=SEED, decreasing_factor=config.decreasing_factor,
                                                    model_type=model_type, alpha=config.alpha, gamma=config.gamma)

n_items = trainloader.n_items
config.p_dims.append(n_items)
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


def train(dataloader, epoch, optimizer):
    if config.data_loader_type in {"stream", "2stages"}:
        dataloader._init_batches()

    global update_count
    global js_reg

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

        if config.regularizer == "PD":
            y = elu_func(y) + 1
            y = torch.einsum('bi,i -> bi', y, torch_pop)
        elif config.regularizer == "PC":
            denominator = torch_n_items[:x.shape[0]] - x.sum(dim=0)
            n_u = torch.norm(torch.einsum('bi, b -> bi', (torch.einsum('bi, bi -> bi', y, 1 - x)), 1/denominator), \
                             p='fro', dim=0)
            c = torch.einsum('bi, i -> bi', (y*config.reg_weight + (1-reg_weight)), 1/torch_abs_pop)
            m_u = torch.norm(torch.einsum('bi, b -> bi', torch.einsum('bi, bi->bi', c, 1-x), 1/denominator), \
                             p='fro', dim=0)
            y = torch.einsum('bi, bi -> bi', y, c*pc_weight*n_u/m_u)


        loss = criterion.log_p(x, y, pos_items=pos_items, neg_items=neg_items, mask=mask, model_type=model_type)

        if config.regularizer == "JS":
            foldin_mask = (x == 1).float()
            topk_scores, topk_items = torch.topk(torch.einsum('bi,bi -> bi', y, foldin_mask), 100)
            topk_pop_distro = torch.einsum('bi, bic -> bc', topk_scores, pop_mapping[topk_items])
            norm_topk_pop_distro = js_div_2d(topk_pop_distro)
            foldin_pop = torch.einsum('bi,ic -> bc', x, pop_mapping)
            foldin_pop = foldin_pop.float()
            norm_foldin_pop = js_div_2d(foldin_pop)
            loss += js_div_2d(norm_topk_pop_distro.unsqueeze(0).unsqueeze(0) + 1e-10,
                              norm_foldin_pop.unsqueeze(0).unsqueeze(0) + 1e-10)[0, 0] * reg_weight
        elif config.regularizer == "boratto":
            # computing the absolute correlation
            vx = loss - torch.mean(loss)
            vy = torch_pop[pos_items.long()] - torch.mean(torch_pop[pos_items.long()])
            correlation = torch.abs(
                torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2))))
            loss = torch.sum(loss)
            loss += correlation * reg_weight
        else:
            loss = torch.sum(loss)

        loss += anneal * criterion.kld(mu, logvar)

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

    return train_loss_cumulative / (1 + batch_idx)


top_k = (1, 5, 10)


def evaluate(dataloader, popularity, tag='validation'):
    global js_reg
    # Turn on evaluation mode
    model.eval()
    accumulator = MetricAccumulator()
    result = collections.defaultdict(float)
    batch_num = 0
    n_users_train = 0
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

            if config.regularizer == "PD":
                y = elu_func(y) + 1

            # print("[EVAL]: model forward done")
            #            loss = criterion(recon_batch, x_input, pos, neg, mask, mask, mu, logvar)
            loss = criterion.log_p(x_input, y, pos_items=pos, neg_items=neg, mask=mask, model_type=model_type)

            if config.regularizer == "JS":
                foldin_mask = (x_input == 1).float()
                topk_scores, topk_items = torch.topk(torch.einsum('bi,bi -> bi', y, foldin_mask), 100)
                topk_pop_distro = torch.einsum('bi, bic -> bc', topk_scores, pop_mapping[topk_items])
                norm_topk_pop_distro = normalize_distr(topk_pop_distro)
                foldout_pop = torch.einsum('bi,ic -> bc', mask_te, pop_mapping)
                foldout_pop = foldout_pop.float()
                norm_foldout_pop = normalize_distr(foldout_pop)
                loss += js_div_2d(norm_topk_pop_distro.unsqueeze(0).unsqueeze(0) + 1e-10,
                                  norm_foldout_pop.unsqueeze(0).unsqueeze(0) + 1e-10)[0, 0] * js_reg
            elif config.regularizer == "boratto":
                # computing the absolute correlation
                vx = loss - torch.mean(loss)
                vy = torch_pop[pos.long()] - torch.mean(torch_pop[pos.long()])
                correlation = torch.abs(
                    torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2))))
                loss = torch.sum(loss)
                loss += correlation * reg_weight
            else:
                loss = torch.sum(loss)

            # loss += 0.*criterion.kld(mu, logvar) # no kld in evaluation
            # print("[EVAL]: Loss computed")
            if np.isinf(loss.item()):
                print("Val Loss inf")
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

# need to init the model prior to the dataloader to build the two-stages stream dataloader
print(config.p_dims)
model = MultiVAE(config.p_dims)
model = model.to(device)

criterion = rvae_rank_pair_loss(device=device, popularity=popularity if model_type in (model_types.LOW, model_types.MED,
                                                                                       model_types.HIGH) else None,
                                scale=config.scale,
                                thresholds=thresholds,
                                frequencies=frequencies)

if config.data_loader_type in {"stream", "2stages"}:
    trainloader.model = model

best_loss = np.Inf

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
        LOW, MED, HIGH = 0, 1, 2
        if model_type in (model_types.BASELINE, model_types.REWEIGHTING, model_types.OVERSAMPLING, model_types.U_SAMPLING):
            val_result = result["loss"]
            if np.isinf(val_result) or np.isnan(val_result):
                val_result = -float(result[f"luciano_stat@{config.best_model_k_metric}"])
        # in the following elif blocks, the value is multiplied by -1 to invert the objective (minimizing instead of
        # maximizing)
        elif model_type == model_types.LOW:
            val_result = -float(result[f"luciano_stat_by_pop@{config.best_model_k_metric}"].split(",")[LOW])
        elif model_type == model_types.MED:
            val_result = -float(result[f"luciano_stat_by_pop@{config.best_model_k_metric}"].split(",")[MED])
        elif model_type == model_types.HIGH:
            val_result = -float(result[f"luciano_stat_by_pop@{config.best_model_k_metric}"].split(",")[HIGH])
        if val_result < best_loss:
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
ax1.set_yscale('log')
ax1.set_title('Train')

ax2.plot(lossTest, color='r')
ax2.set_yscale('log')
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
result_validation = evaluate(trainloader, popularity, 'validation')

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
    fp.write(f'Data_Loader_Type = {config.data_loader_type}\n')
    fp.write(f"Regularizer = {config.regularizer}\n")
    fp.write(f"Dataset = {dataset_name}\n")
    fp.write(f"Seed = {SEED}\n")
    fp.write('\n')

    for k in config.__dict__:
        fp.write(f'{k} = {config.__dict__[k]}\n')

    fp.write(f"Average_Exposure = {np.mean(trainloader.item_visibility_dict['training'])}\n")
    fp.write(f"Std_Exposure = {np.std(trainloader.item_visibility_dict['training'])}\n")
    if model_type == model_types.OVERSAMPLING:
        fp.write(f"Multiplier for oversampling formula = {trainloader.h}\n")
    try:
        fp.write(f"Contamination = {trainloader.contamination}\n")
    except:
        pass


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
