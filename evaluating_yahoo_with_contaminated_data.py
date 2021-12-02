import collections
import os
from datetime import datetime
import torch
import numpy as np
import json
from evaluation import MetricAccumulator
from util import compute_max_y_aux_popularity, naive_sparse2tensor, set_seed
from models import MultiVAE
from loss_func import rvae_rank_pair_loss
from data_loaders import DataLoader, CachedDataLoader
from config import Config
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-model_type", help="model type to evaluate",
                    type=str, default="model_types.BASELINE")
args = parser.parse_args()

datasets = Config("./datasets_info.json")
model_types = Config("./model_type_info.json")

# SETTINGS ------------------------------------------
config = Config("./rvae_config.json")

if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = config.CUDA_VISIBLE_DEVICES

dataset_name = eval("datasets.YAHOO")
model_type = eval(args.model_type)
copy_pasting_data = True
config.cached_dataloader = False
config.metrics_scale = eval("1/15")
config.use_popularity = True
config.p_dims = eval("[32, 600]")
config.alpha = 0.02
config.gamma = 10.
if model_type == "reweighting":
    assert config.alpha > 0 and config.gamma > 0, config.alpha

# ---------------------------------------------------

# set seed for experiment reproducibility
SEED = 12121995
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
dataset_file = os.path.join(data_dir, 'contaminated_data_rvae')

result_dir = os.path.join(data_dir, 'results')
if not os.path.exists(result_dir):
    os.makedirs(result_dir)



""" Train and test"""

if not config.cached_dataloader:
    if model_type == model_types.BASELINE:
        trainloader = DataLoader(dataset_file, seed=SEED, decreasing_factor=1, model_type=model_type)
    else:
        trainloader = DataLoader(dataset_file, seed=SEED, decreasing_factor=config.decreasing_factor,
                                 model_type=model_type, alpha=config.alpha, gamma=config.gamma)
else:
    print('USE CACHED DATALOADER')
    if model_type == model_types.BASELINE:
        trainloader = CachedDataLoader(dataset_file, seed=SEED, decreasing_factor=1, model_type=model_type)
    else:
        trainloader = CachedDataLoader(dataset_file, seed=SEED, decreasing_factor=config.decreasing_factor,
                                 model_type=model_type, alpha=config.alpha, gamma=config.gamma)

run_time = datetime.today().strftime('%Y%m%d_%H%M')
run_dir = os.path.join(result_dir, f'{model_type}_ncontaminated_{trainloader.contamination}_{run_time}')

os.makedirs(run_dir, exist_ok=True)

# TODO: include learning params
file_model = os.path.join(run_dir, 'best_model.pth')

to_pickle = True

n_items = trainloader.n_items
config.p_dims.append(n_items)
popularity = trainloader.item_popularity_dict["training"]
thresholds = trainloader.thresholds
frequencies = trainloader.frequencies_dict["training"]

max_y_aux_popularity = compute_max_y_aux_popularity(config)

top_k = (1, 5, 10)
criterion = rvae_rank_pair_loss(device=device, popularity=popularity if model_type in (model_types.LOW, model_types.MED,
                                                                model_types.HIGH) else None,
                                scale=config.scale,
                                thresholds=thresholds,
                                frequencies=frequencies)

def evaluate(dataloader, popularity, tag='validation'):
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


model = MultiVAE(config.p_dims)
model = model.to(device)
saved_model = os.path.join(data_dir, model_type, "best_model.pth")
model.load_state_dict(torch.load(saved_model, map_location=device))
model.eval()


result_test = evaluate(trainloader, popularity, 'test')
renaming_luciano_stat = {"loss": "loss", "train_loss": "train_loss"}
d1 = {f"luciano_recalled_by_pop@{k}": f"recall_by_pop@{k}" for k in [1, 5, 10]}
d2 = {f"luciano_stat_by_pop@{k}": f"hit_rate_by_pop@{k}" for k in [1, 5, 10]}
d3 = {f"luciano_stat@{k}": f"hit_rate@{k}" for k in [1, 5, 10]}
d4 = {f"luciano_weighted_stat@{k}": f"weighted_hit_Rate@{k}" for k in [1, 5, 10]}
renaming_luciano_stat = {**renaming_luciano_stat, **d1, **d2, **d3, **d4}

print(f'K = {config.gamma_k}')
print("Test Statistics: \n")
print('\n'.join([f'{renaming_luciano_stat[k]:<23}{v}' for k, v in sorted(result_test.items())
                 if k in renaming_luciano_stat]))

"""# Save result"""
# all results
def renaming_results(result_dict, rename_dict):
    return {rename_dict[k]: v for k, v in result_dict.items() if k in rename_dict}

# test results
with open(os.path.join(run_dir, 'result_test.json'), 'w') as fp:
    json.dump(renaming_results(result_test, renaming_luciano_stat), fp,
              indent=4, sort_keys=True)

print('DONE')
print(run_dir)