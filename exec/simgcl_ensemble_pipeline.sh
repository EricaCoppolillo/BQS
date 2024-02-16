# pick dataset(s) of interest - MOVIELENS_1M CITEULIKE PINTEREST YAHOO AMAZON_GGF
declare -a datasets=("MOVIELENS_1M" "PINTEREST" "YAHOO" "CITEULIKE" "AMAZON_GGF")
declare -A datasets_dict=(["MOVIELENS_1M"]='ml-1m' ["PINTEREST"]='pinterest' ["YAHOO"]='yahoo-r3' ["CITEULIKE"]='citeulike' ["AMAZON_GGF"]='amzn-ggf')
declare -A deltas_dict=( ["MOVIELENS_1M"]="0.40" ["PINTEREST"]="0.60" ["YAHOO"]="0.55" ["AMAZON_GGF"]="0.20" ["CITEULIKE"]="0.70")

cuda="1"
# moving in the parent directory
cd ../
declare -a seeds=(12121995 230782 190291 81163 100362)

simgcl_config_file="simgcl_config.json"
ensemble_config_file="ensemble_config.json"

for seed in "${seeds[@]}"; do
  # iterating across datasets
  for dataset_name in "${datasets[@]}"; do
    jsonStr=$(cat $simgcl_config_file)
    jq '.seed = '$seed <<<"$jsonStr" > $simgcl_config_file
    jsonStr=$(cat $simgcl_config_file)
    jq '.copy_pasting_data = "True"' <<<"$jsonStr" > $simgcl_config_file
    jsonStr=$(cat $simgcl_config_file)
    jq '.dataset_name = "datasets.'$dataset_name'"' <<<"$jsonStr" > $simgcl_config_file
    jsonStr=$(cat $simgcl_config_file)
    jq '.model_type = "model_types.BASELINE"' <<<"$jsonStr" > $simgcl_config_file
    jsonStr=$(cat $simgcl_config_file)
    jq '.regularizer = ""' <<<"$jsonStr" > $simgcl_config_file
    jsonStr=$(cat $simgcl_config_file)
    jq '.data_loader_type = ""' <<<"$jsonStr" > $simgcl_config_file
    jsonStr=$(cat $simgcl_config_file)
    jq '.CUDA_VISIBLE_DEVICES = "'$cuda'"' <<<"$jsonStr" > $simgcl_config_file
    echo "Current Dataset: "$dataset_name

    # training the baseline
    jsonStr=$(cat $simgcl_config_file)
    jq '.dir_name = "baseline_'$seed'"' <<<"$jsonStr" > $simgcl_config_file

    jsonStr=$(cat $simgcl_config_file)
    jq '.model_type = "model_types.LOW"' <<<"$jsonStr" > $simgcl_config_file

    # training the low model
    jsonStr=$(cat $simgcl_config_file)
    jq '.dir_name = "low_'$seed'"' <<<"$jsonStr" > $simgcl_config_file

    # copy-paste baseline model and low model of the specific seed into the "main" baseline and low folders
    dataset_new_fmt="${datasets_dict[$dataset_name]}"
    root_path='/mnt/nas/minici/RVAE_SIGIR23/data/'$dataset_new_fmt
    root_baseline_path=$root_path'/baseline'
    root_low_path=$root_path'/low'
    path_to_baseline=$root_path'/simGCL/results/baseline_'$seed
    path_to_low=$root_path'/simGCL/results/low_'$seed
    cp -rf $path_to_baseline $root_baseline_path
    cp -rf $path_to_low $root_low_path

    for ensemble_weight in $(seq 0.05 0.05 1.0); do
      echo "Evaluating Ensemble..."
      new_ensemble_weight="${ensemble_weight//,/.}"
      echo $new_ensemble_weight
      # new_ensemble_weight="${deltas_dict[$dataset_name]}"
      jsonStr=$(cat ensemble_config.json)
      jq '.ensemble_weight = '$new_ensemble_weight <<<"$jsonStr" > ensemble_config.json
      jsonStr=$(cat ensemble_config.json)
      jq '.seed = '$seed <<<"$jsonStr" > ensemble_config.json
      jsonStr=$(cat ensemble_config.json)
      jq '.algorithm = "simGCL"' <<<"$jsonStr" > $ensemble_config_file
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
      sleep 60s
      done
  done
done