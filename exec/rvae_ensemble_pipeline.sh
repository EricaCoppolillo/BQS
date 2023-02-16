# pick dataset(s) of interest - MOVIELENS_1M CITEULIKE PINTEREST YAHOO AMAZON_GGF
declare -a datasets=("MOVIELENS_1M" "PINTEREST" "YAHOO" "AMAZON_GGF" "CITEULIKE")

cuda="1"
# moving in the parent directory
cd ../
declare -a seeds=(12121995 230782 190291 81163 100362)

ensemble_config_file="ensemble_config.json"

for seed in "${seeds[@]}"; do
  # iterating across datasets
  for dataset_name in "${datasets[@]}"; do
    jsonStr=$(cat rvae_config.json)
    jq '.seed = '$seed <<<"$jsonStr" > rvae_config.json
    jsonStr=$(cat rvae_config.json)
    jq '.copy_pasting_data = "True"' <<<"$jsonStr" > rvae_config.json
    jsonStr=$(cat rvae_config.json)
    jq '.dataset_name = "datasets.'$dataset_name'"' <<<"$jsonStr" > rvae_config.json
    jsonStr=$(cat rvae_config.json)
    jq '.model_type = "model_types.BASELINE"' <<<"$jsonStr" > rvae_config.json
    jsonStr=$(cat rvae_config.json)
    jq '.regularizer = ""' <<<"$jsonStr" > rvae_config.json
    jsonStr=$(cat rvae_config.json)
    jq '.data_loader_type = ""' <<<"$jsonStr" > rvae_config.json
    jsonStr=$(cat rvae_config.json)
    jq '.CUDA_VISIBLE_DEVICES = "'$cuda'"' <<<"$jsonStr" > rvae_config.json
    echo "Current Dataset: "$dataset_name

    # training the baseline
    jsonStr=$(cat rvae_config.json)
    jq '.dir_name = "baseline_'$seed'"' <<<"$jsonStr" > rvae_config.json
    python3 rvae.py
    jsonStr=$(cat rvae_config.json)
    jq '.model_type = "model_types.LOW"' <<<"$jsonStr" > rvae_config.json

    # training the low model
    jsonStr=$(cat rvae_config.json)
    jq '.dir_name = "ensemble_'$seed'"' <<<"$jsonStr" > rvae_config.json
    python3 rvae.py

    for ensemble_weight in $(seq 0.05 0.05 1.0); do
      sleep 60s
      echo "Evaluating Ensemble..."
      new_ensemble_weight="${ensemble_weight//,/.}"
      jsonStr=$(cat ensemble_config.json)
      jq '.ensemble_weight = '$new_ensemble_weight <<<"$jsonStr" > ensemble_config.json
      jsonStr=$(cat ensemble_config.json)
      jq '.seed = '$seed <<<"$jsonStr" > ensemble_config.json
      jsonStr=$(cat ensemble_config.json)
      jq '.algorithm = "rvae"' <<<"$jsonStr" > $ensemble_config_file
      jsonStr=$(cat $ensemble_config_file)
      jq '.dataset_name = "datasets.'$dataset_name'"' <<<"$jsonStr" > ensemble_config.json
      jsonStr=$(cat ensemble_config.json)
      jq '.model_type = "simple"' <<<"$jsonStr" > ensemble_config.json
      jsonStr=$(cat ensemble_config.json)
      jq '.CUDA_VISIBLE_DEVICES = "'$cuda'"' <<<"$jsonStr" > ensemble_config.json
      jsonStr=$(cat ensemble_config.json)
      jq '.dir_name = "ensemble_'$seed'_'$new_ensemble_weight'"' <<<"$jsonStr" > ensemble_config.json
      # running the ensemble model
      python3 ensemble.py $ensemble_config_file
    done
  done
done