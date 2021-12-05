# pick dataset(s) of interest - MOVIELENS_1M MOVIELENS_20M CITEULIKE PINTEREST EPINIONS NETFLIX
declare -a datasets=("MOVIELENS_1M" "PINTEREST" "CITEULIKE")
declare -A datasets_path_name=( ["MOVIELENS_1M"]="ml-1m" ["MOVIELENS_20M"]="ml-20m" ["CITEULIKE"]="citeulike-a" ["PINTEREST"]="pinterest" ["EPINIONS"]="epinions" ["NETFLIX"]="netflix")
declare -A dataset2alpha=( ["MOVIELENS_1M"]="0.01" ["MOVIELENS_20M"]="0.0025" ["CITEULIKE"]="0.05" ["PINTEREST"]="0.05" ["EPINIONS"]="0.1" ["NETFLIX"]="0.0005")
declare -A dataset2gamma=( ["MOVIELENS_1M"]="53.33" ["MOVIELENS_20M"]="10" ["CITEULIKE"]="6.67" ["PINTEREST"]="10" ["EPINIONS"]="6.67" ["NETFLIX"]="10")

cuda="1"
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
    jq '.dataset_name = "datasets."'$dataset_name <<<"$jsonStr" > rvae_config.json
    jq '.model_type = "model_types.BASELINE"' <<<"$jsonStr" > rvae_config.json
    jq '.regularizer = ""' <<<"$jsonStr" > rvae_config.json
    jq '.data_loader_type = ""' <<<"$jsonStr" > rvae_config.json
    jq '.CUDA_VISIBLE_DEVICES = '$cuda <<<"$jsonStr" > rvae_config.json
    python rvae.py
    # oversampling
    jq '.model_type = "model_types.OVERSAMPLING"' <<<"$jsonStr" > rvae_config.json
    python rvae.py
    # reweighting
    jq '.model_type = "model_types.REWEIGHTING"' <<<"$jsonStr" > rvae_config.json
    jq '.alpha = '$dataset2alpha[$dataset_name] <<<"$jsonStr" > rvae_config.json
    jq '.gamma = 'dataset2gamma[$dataset_name] <<<"$jsonStr" > rvae_config.json
    python rvae.py
    # Competitors
    # jannach
    for jannach_weight in $(seq 1.5 0.25 3.0); do
      rm -rf ./data/$datasets_path_name[$dataset_name]/preprocessed_data/baselinejannach
      rm -rf ./data/$datasets_path_name[$dataset_name]/sparse_matrices/baselinejannach
      jq '.model_type = "model_types.BASELINE"' <<<"$jsonStr" > rvae_config.json
      jq '.jannach_width = '$jannach_weight <<<"$jsonStr" > rvae_config.json
      jq '.regularizer = ""' <<<"$jsonStr" > rvae_config.json
      jq '.data_loader_type = "jannach"' <<<"$jsonStr" > rvae_config.json
      python rvae.py
    done
    # ips
    rm -rf ./data/$datasets_path_name[$dataset_name]/preprocessed_data/reweighting
    rm -rf ./data/$datasets_path_name[$dataset_name]/sparse_matrices/reweighting
    jq '.model_type = "model_types.REWEIGHTING"' <<<"$jsonStr" > rvae_config.json
    jq '.alpha = "None"' <<<"$jsonStr" > rvae_config.json
    jq '.gamma = "None"' <<<"$jsonStr" > rvae_config.json
    python rvae.py
    # boratto sampling
    # rm -rf ./data/$datasets_path_name[$dataset_name]/preprocessed_data/baselineboratto
    # rm -rf ./data/$datasets_path_name[$dataset_name]/sparse_matrices/baselineboratto
    jq '.model_type = "model_types.BASELINE"' <<<"$jsonStr" > rvae_config.json
    jq '.data_loader_type = "boratto"' <<<"$jsonStr" > rvae_config.json
    python rvae.py
    # boratto reweighting
    jq '.data_loader_type = ""' <<<"$jsonStr" > rvae_config.json
    for boratto_reg in $(seq 0.1 0.1 1.5); do
      jq '.model_type = "model_types.BASELINE"' <<<"$jsonStr" > rvae_config.json
      jq '.regularizer = '$boratto_reg <<<"$jsonStr" > rvae_config.json
      python rvae.py
    done
    # boratto sampling + reweighting
    jq '.data_loader_type = "boratto"' <<<"$jsonStr" > rvae_config.json
    for boratto_reg in $(seq 0.1 0.1 1.5); do
      jq '.model_type = "model_types.BASELINE"' <<<"$jsonStr" > rvae_config.json
      jq '.regularizer = '$boratto_reg <<<"$jsonStr" > rvae_config.json
      python rvae.py
    done
    # TODO: method based on causal inference
done