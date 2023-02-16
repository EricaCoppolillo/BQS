# pick dataset(s) of interest - MOVIELENS_1M CITEULIKE PINTEREST YAHOO AMAZON_GGF
declare -a datasets=("PINTEREST" "AMAZON_GGF")

cuda="1"
bpr_config_file="bpr_config.json"
ensemble_config_file="ensemble_config.json"

# moving in the parent directory
cd ../
declare -a seeds=(12121995 190291 230782 81163 100362)

for seed in "${seeds[@]}"; do
  for dataset_name in "${datasets[@]}"; do
    # baseline
    jsonStr=$(cat $bpr_config_file)
    jq '.seed = '$seed <<<"$jsonStr" > $bpr_config_file
    jsonStr=$(cat $bpr_config_file)
    jq '.dataset_name = "datasets.'$dataset_name'"' <<<"$jsonStr" > $bpr_config_file
    jsonStr=$(cat $bpr_config_file)
    jq '.model_type = "model_types.LOW"' <<<"$jsonStr" > $bpr_config_file
    jsonStr=$(cat $bpr_config_file)
    jq '.cached_dataloader = "True"' <<<"$jsonStr" > $bpr_config_file
    jsonStr=$(cat $bpr_config_file)
    jq '.n_epochs = 20' <<<"$jsonStr" > $bpr_config_file
    jsonStr=$(cat $bpr_config_file)
    jq '.regularizer = ""' <<<"$jsonStr" > $bpr_config_file
    jsonStr=$(cat $bpr_config_file)
    jq '.data_loader_type = ""' <<<"$jsonStr" > $bpr_config_file
    jsonStr=$(cat $bpr_config_file)
    jq '.CUDA_VISIBLE_DEVICES = "'$cuda'"' <<<"$jsonStr" > $bpr_config_file
    jsonStr=$(cat $bpr_config_file)
    jq '.dir_name = "low_'$seed'"' <<<"$jsonStr" > $bpr_config_file
    python3 bpr.py $bpr_config_file

#    Low for Ensemble
    jsonStr=$(cat $bpr_config_file)
    jq '.model_type = "model_types.BASELINE"' <<<"$jsonStr" > $bpr_config_file
    jsonStr=$(cat $bpr_config_file)
    jq '.cached_dataloader = "False"' <<<"$jsonStr" > $bpr_config_file
    jsonStr=$(cat $bpr_config_file)
    jq '.dir_name = "newbaseline_'$seed'"' <<<"$jsonStr" > $bpr_config_file
    # training the low model
    python3 bpr.py $bpr_config_file

#     Ensemble
    for ensemble_weight in $(seq 0.00005 0.00005 0.00045); do
      sleep 60s
      new_ensemble_weight="${ensemble_weight//,/.}"
      jsonStr=$(cat $ensemble_config_file)
      jq '.ensemble_weight = '$new_ensemble_weight <<<"$jsonStr" > $ensemble_config_file
      jsonStr=$(cat $ensemble_config_file)
      jq '.latent_dim = "32"' <<<"$jsonStr" > $ensemble_config_file
      jsonStr=$(cat $ensemble_config_file)
      jq '.algorithm = "bpr"' <<<"$jsonStr" > $ensemble_config_file
      jsonStr=$(cat $ensemble_config_file)
      jq '.seed = '$seed <<<"$jsonStr" > $ensemble_config_file
      jsonStr=$(cat $ensemble_config_file)
      jq '.dataset_name = "datasets.'$dataset_name'"' <<<"$jsonStr" > $ensemble_config_file
      jsonStr=$(cat $ensemble_config_file)
      jq '.model_type = "simple"' <<<"$jsonStr" > $ensemble_config_file
      jsonStr=$(cat $ensemble_config_file)
      jq '.CUDA_VISIBLE_DEVICES = "'$cuda'"' <<<"$jsonStr" > $ensemble_config_file
      jsonStr=$(cat $ensemble_config_file)
      jq '.dir_name = "newensemble_'$seed'_'$new_ensemble_weight'"' <<<"$jsonStr" > $ensemble_config_file
      # running the ensemble model
      python3 ensemble.py $ensemble_config_file
    done
  done
done