# pick dataset(s) of interest - MOVIELENS_1M CITEULIKE PINTEREST YAHOO AMAZON_GGF
declare -a datasets=("MOVIELENS_1M" "PINTEREST" "YAHOO" "CITEULIKE" "AMAZON_GGF")
declare -A deltas_dict=( ["MOVIELENS_1M"]="0.40" ["PINTEREST"]="0.60" ["YAHOO"]="0.55" ["AMAZON_GGF"]="0.20" ["CITEULIKE"]="0.70")

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
    # python3 rvae.py

    jsonStr=$(cat rvae_config.json)
    jq '.model_type = "model_types.LOW"' <<<"$jsonStr" > rvae_config.json

    # training the low model
    jsonStr=$(cat rvae_config.json)
    jq '.dir_name = "low_'$seed'"' <<<"$jsonStr" > rvae_config.json
    # python3 rvae.py

#    for ensemble_weight in $(seq 0.05 0.05 1.0); do
#      echo "Evaluating Ensemble..."
#      new_ensemble_weight="${ensemble_weight//,/.}"
#      echo $new_ensemble_weight
    new_ensemble_weight="${deltas_dict[$dataset_name]}"
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
    jq '.dir_name = "newensemble_'$seed'_'$new_ensemble_weight'"' <<<"$jsonStr" > ensemble_config.json
    # running the ensemble model
    python3 ensemble.py $ensemble_config_file
#      sleep 60s
#    done
  done
done