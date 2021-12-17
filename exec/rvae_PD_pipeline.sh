# pick dataset(s) of interest - MOVIELENS_1M MOVIELENS_20M CITEULIKE PINTEREST EPINIONS NETFLIX
declare -a datasets=("MOVIELENS_1M" "PINTEREST" "CITEULIKE")
declare -A datasets_path_name=( ["MOVIELENS_1M"]="ml-1m" ["MOVIELENS_20M"]="ml-20m" ["CITEULIKE"]="citeulike-a" ["PINTEREST"]="pinterest" ["EPINIONS"]="epinions" ["NETFLIX"]="netflix")

cuda=""
# moving in the parent directory
cd ../
declare -a seeds=(12121995 230782 190291 81163 100362)

for seed in "${seeds[@]}"; do

  for dataset_name in "${datasets[@]}"; do
    # deleting the data directory
    # rm -rf ./data/$dataset_name/preprocessed_data/baseline
    # baseline
    jsonStr=$(cat rvae_config.json)
    jq '.seed = '$seed <<<"$jsonStr" > rvae_config.json
    jsonStr=$(cat rvae_config.json)
    jq '.dataset_name = "datasets.'$dataset_name'"' <<<"$jsonStr" > rvae_config.json
    jsonStr=$(cat rvae_config.json)
    jq '.data_loader_type = ""' <<<"$jsonStr" > rvae_config.json
    jsonStr=$(cat rvae_config.json)
    jq '.CUDA_VISIBLE_DEVICES = "'$cuda'"' <<<"$jsonStr" > rvae_config.json
    # Competitors
    # CAUSAL PD
    jsonStr=$(cat rvae_config.json)
    jq '.data_loader_type = ""' <<<"$jsonStr" > rvae_config.json
    for pd_reg in $(seq 0.02 0.02 0.25); do
      pd_reg="${pd_reg//,/.}"
      jsonStr=$(cat rvae_config.json)
      jq '.model_type = "model_types.BASELINE"' <<<"$jsonStr" > rvae_config.json
      jsonStr=$(cat rvae_config.json)
      jq '.reg_weight = "'$pd_reg'"' <<<"$jsonStr" > rvae_config.json
      jsonStr=$(cat rvae_config.json)
      jq '.regularizer = "PD"' <<<"$jsonStr" > rvae_config.json
      python rvae.py
    done
  done
done