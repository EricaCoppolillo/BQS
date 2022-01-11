# pick dataset(s) of interest - MOVIELENS_1M MOVIELENS_20M CITEULIKE PINTEREST EPINIONS NETFLIX YAHOO COAT AMAZON_GGF
declare -a datasets=("YAHOO")
declare -A datasets_path_name=( ["MOVIELENS_1M"]="ml-1m" ["MOVIELENS_20M"]="ml-20m" ["CITEULIKE"]="citeulike-a" ["PINTEREST"]="pinterest" ["EPINIONS"]="epinions" ["NETFLIX"]="netflix", ["YAHOO"]="yahoo-r3", ["COAT"]="coat",  ["AMAZON_GGF"]="amzn-ggf")
cuda="2"
# moving in the parent directory
cd ../
declare -a seeds=(12121995 230782 190291 81163 100362)

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
    if [ $seed != 12121995 ]; then
      echo "Training Baseline and Low Models"
      sleep 10s
      # training the baseline
      python rvae.py
      jsonStr=$(cat rvae_config.json)
      jq '.model_type = "model_types.LOW"' <<<"$jsonStr" > rvae_config.json
      # training the low model
      python rvae.py
    fi
    
    for ensemble_weight in $(seq 0.05 0.05 1.0); do
      echo "Evaluating Ensemble..."
      new_ensemble_weight="${ensemble_weight//,/.}"
      jsonStr=$(cat ensemble_config.json)
      jq '.ensemble_weight = '$new_ensemble_weight <<<"$jsonStr" > ensemble_config.json
      jsonStr=$(cat ensemble_config.json)
      jq '.seed = '$seed <<<"$jsonStr" > ensemble_config.json
      jsonStr=$(cat ensemble_config.json)
      jq '.dataset_name = "datasets.'$dataset_name'"' <<<"$jsonStr" > ensemble_config.json
      jsonStr=$(cat ensemble_config.json)
      jq '.model_type = "simple"' <<<"$jsonStr" > ensemble_config.json
      jsonStr=$(cat ensemble_config.json)
      jq '.CUDA_VISIBLE_DEVICES = "'$cuda'"' <<<"$jsonStr" > ensemble_config.json
      # running the ensemble model
      python ensemble.py
    done
  done
done