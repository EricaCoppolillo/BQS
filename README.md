# Welcome to the official repository of Conservative Popularity Debiasing in Pairwise Ranking for Recommendation

## Reproducibility

To be able to effectively reproduce our results, follow these steps:
1. Create a conda environment using the provided .yml. Execute `conda env create -f environment.yml` on your command-line tool.
2. Execute `cd exec && ./rvae_pipeline.sh` to reproduce RVAE experiments and `cd exec && ./bpr_pipeline.sh` for BPR. 

## Single Experiment

If you wish to execute single runs, customize `rvae_config.json` (resp. `bpr_config.json`) with the following parameters:
  - "CUDA_VISIBLE_DEVICES": the Cuda GPU ID if you have available one, otherwise an empty string to use CPU.
  - "dataset_name": dataset on which you want to adopt, for the names have a look at the `datasets_info.json` file.
  - "model_type": for our methods pick in \{BASELINE, OVERSAMPLING\}. For competitors, use `model_types.BASELINE`.
  - "copy_pasting_data": either "True" or "False" to copy the results in the main folder (minor parameter).
  - "latent_dim": only present for BPR. It represents the latent dimension of the embeddings.
  - "cached_dataloader": if you have problems handling the computation in-memory, flag this parameter as "True". You will use an implementation that uses disk allocation. Main risk: it is much slower.
  - "data_loader_type": pick one in \{"", "jannach", "boratto"\}. The first is the standard data loader, "jannach" and "boratto" sample following the strategies depicted in the omonymous papers.
  - "regularizer: pick one in \{"", "boratto", "PD"\}. The first means no regularization, "boratto" uses a penalty consisting of the correlation between the loss residuals and the positive item's popularity, and "PD" adds up the popularity factor during the training stage.
  - "reg_weight": is the coefficient that weights the contribution of the regularization.
  - "seed": it sets the seed to obtain reproducible results.
  - there are also other parameters (e.g.: n_epochs, batch_size) that are straightforward to interpret.

After having set `rvae_config.json` (resp. `bpr_config.json`), execute `python3 rvae.py rvae_config.json` (resp. `python3 bpr.py bpr_config.json`)

### Ensemble configuration

If you wish to execute the ensemble, follow these steps:
  - Execute a run of RVAE (resp. BPR) setting "model_type" parameter equal to "model_types.LOW" and "copy_pasting_data" equal to "True".
  - Repeat the same, but sets "model_type" equal to "model_types.BASELINE".
  - Configure the `ensemble_config.json` by properly setting the "algorithm" field to either "rvae" or "bpr" and "ensemble_weight" to the weight you wish to use.
  - Execute `python3 ensemble.py ensemble_config.json`

### Find the best ensemble configuration

To compute the best delta-score value, hence to find the best ensemble configuration for each dataset:
  - Execute `python3 search_best_ensemble_delta.py $algorithm`, specifying `rvae` or `bpr` instead of `$algorithm`.
  - The script will generate a file named `$algorithm_best_ensemble_delta.csv` in the `data` folder.

### Average the results

To obtain the final results, averaging the five single runs per dataset:
  - Execute `python3 average_runs.py $algorithm`, specifying `rvae` or `bpr` instead of `$algorithm`.
  - The script will generate two files named `$algorithm_$strategy_[avg|psi_avg]_test_results.json` in `data/$dataset/results` folder, representing the results in terms of all the metrics or psi only, respectively.

