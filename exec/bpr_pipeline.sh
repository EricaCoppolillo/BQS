# pick dataset(s) of interest - MOVIELENS_1M MOVIELENS_20M CITEULIKE PINTEREST EPINIONS NETFLIX YAHOO COAT AMAZON_GGF
declare -a datasets=("YAHOO")
declare -A datasets_path_name=( ["MOVIELENS_1M"]="ml-1m" ["MOVIELENS_20M"]="ml-20m" ["CITEULIKE"]="citeulike-a" ["PINTEREST"]="pinterest" ["EPINIONS"]="epinions" ["NETFLIX"]="netflix", ["YAHOO"]="yahoo-r3", ["COAT"]="coat",  ["AMAZON_GGF"]="amzn-ggf")
declare -A dataset2alpha=( ["MOVIELENS_1M"]="0.01" ["MOVIELENS_20M"]="0.0025" ["CITEULIKE"]="0.05" ["PINTEREST"]="0.05" ["EPINIONS"]="0.1" ["NETFLIX"]="0.0005", ["YAHOO"]="0.02", ["COAT"]="0.5",  ["AMAZON_GGF"]="0.01")
declare -A dataset2gamma=( ["MOVIELENS_1M"]="53.33" ["MOVIELENS_20M"]="10" ["CITEULIKE"]="6.67" ["PINTEREST"]="10" ["EPINIONS"]="6.67" ["NETFLIX"]="10", ["YAHOO"]="10", ["COAT"]="7.0",  ["AMAZON_GGF"]="23.33")
declare -A dataset2latent_dim=( ["MOVIELENS_1M"]="64" ["MOVIELENS_20M"]="10" ["CITEULIKE"]="128" ["PINTEREST"]="128" ["EPINIONS"]="6.67" ["NETFLIX"]="10", ["YAHOO"]="32", ["COAT"]="7.0",  ["AMAZON_GGF"]="256")


cuda="2"
bpr_config_file="bpr_config.json"
ensemble_config_file="ensemble_config.json"

# moving in the parent directory
cd ../
declare -a seeds=(12121995 190291 230782 81163 100362)

for seed in "${seeds[@]}"; do

  for dataset_name in "${datasets[@]}"; do
    # deleting the data directory
    # rm -rf ./data/$dataset_name/preprocessed_data/baseline
    # baseline
    jsonStr=$(cat $bpr_config_file)
    jq '.seed = '$seed <<<"$jsonStr" > $bpr_config_file
    jsonStr=$(cat $bpr_config_file)
    # jq '.latent_dim = "'${dataset2latent_dim[$dataset_name]}'"' <<<"$jsonStr" > $bpr_config_file
    # jsonStr=$(cat $bpr_config_file)
    jq '.dataset_name = "datasets.'$dataset_name'"' <<<"$jsonStr" > $bpr_config_file
    jsonStr=$(cat $bpr_config_file)
    jq '.model_type = "model_types.BASELINE"' <<<"$jsonStr" > $bpr_config_file
    jsonStr=$(cat $bpr_config_file)
    jq '.regularizer = ""' <<<"$jsonStr" > $bpr_config_file
    jsonStr=$(cat $bpr_config_file)
    jq '.data_loader_type = ""' <<<"$jsonStr" > $bpr_config_file
    jsonStr=$(cat $bpr_config_file)
    jq '.CUDA_VISIBLE_DEVICES = "'$cuda'"' <<<"$jsonStr" > $bpr_config_file
    python bpr.py $bpr_config_file
    # Low for Ensemble
    jsonStr=$(cat $bpr_config_file)
    jq '.model_type = "model_types.LOW"' <<<"$jsonStr" > $bpr_config_file
    # training the low model
    python bpr.py $bpr_config_file
    # Ensemble
    for ensemble_weight in $(seq 0.05 0.05 1.0); do
      new_ensemble_weight="${ensemble_weight//,/.}"
      jsonStr=$(cat $ensemble_config_file)
      jq '.ensemble_weight = '$new_ensemble_weight <<<"$jsonStr" > $ensemble_config_file
      jsonStr=$(cat $ensemble_config_file)
      # jq '.latent_dim = "'${dataset2latent_dim[$dataset_name]}'"' <<<"$jsonStr" > $ensemble_config_file
      # jsonStr=$(cat $ensemble_config_file)
      jq '.algorithm = "bpr"' <<<"$jsonStr" > $ensemble_config_file
      jsonStr=$(cat $ensemble_config_file)
      jq '.seed = '$seed <<<"$jsonStr" > $ensemble_config_file
      jsonStr=$(cat $ensemble_config_file)
      jq '.dataset_name = "datasets.'$dataset_name'"' <<<"$jsonStr" > $ensemble_config_file
      jsonStr=$(cat $ensemble_config_file)
      jq '.model_type = "simple"' <<<"$jsonStr" > $ensemble_config_file
      jsonStr=$(cat $ensemble_config_file)
      jq '.CUDA_VISIBLE_DEVICES = "'$cuda'"' <<<"$jsonStr" > $ensemble_config_file
      # running the ensemble model
      python ensemble.py $ensemble_config_file
    done
    # oversampling
    jsonStr=$(cat $bpr_config_file)
    jq '.model_type = "model_types.OVERSAMPLING"' <<<"$jsonStr" > $bpr_config_file
    python bpr.py $bpr_config_file
    # reweighting
    jsonStr=$(cat $bpr_config_file)
    jq '.model_type = "model_types.REWEIGHTING"' <<<"$jsonStr" > $bpr_config_file
    jsonStr=$(cat $bpr_config_file)
    jq '.alpha = "'${dataset2alpha[$dataset_name]}'"' <<<"$jsonStr" > $bpr_config_file
    jsonStr=$(cat $bpr_config_file)
    jq '.gamma = "'${dataset2gamma[$dataset_name]}'"' <<<"$jsonStr" > $bpr_config_file
    python bpr.py $bpr_config_file
    # Competitors
    # jannach
    for jannach_weight in $(seq 1.5 0.25 3.0); do
      jannach_weight="${jannach_weight//,/.}"
      rm -rf ./data/${datasets_path_name[$dataset_name]}/preprocessed_data/bpr/baselinejannach/decreasing_factor_1/$seed
      rm -rf ./data/${datasets_path_name[$dataset_name]}/sparse_matrices/bpr/baselinejannach/decreasing_factor_1/$seed
      jsonStr=$(cat $bpr_config_file)
      jq '.model_type = "model_types.BASELINE"' <<<"$jsonStr" > $bpr_config_file
      jsonStr=$(cat $bpr_config_file)
      jq '.jannach_width = "'$jannach_weight'"' <<<"$jsonStr" > $bpr_config_file
      jsonStr=$(cat $bpr_config_file)
      jq '.regularizer = ""' <<<"$jsonStr" > $bpr_config_file
      jsonStr=$(cat $bpr_config_file)
      jq '.data_loader_type = "jannach"' <<<"$jsonStr" > $bpr_config_file
      python bpr.py $bpr_config_file
    done
    # ips
    rm -rf ./data/${datasets_path_name[$dataset_name]}/preprocessed_data/bpr/reweighting/decreasing_factor_1/$seed
    rm -rf ./data/${datasets_path_name[$dataset_name]}/sparse_matrices/bpr/reweighting/decreasing_factor_1/$seed
    jsonStr=$(cat $bpr_config_file)
    jq '.data_loader_type = ""' <<<"$jsonStr" > $bpr_config_file
    jsonStr=$(cat $bpr_config_file)
    jq '.model_type = "model_types.REWEIGHTING"' <<<"$jsonStr" > $bpr_config_file
    jsonStr=$(cat $bpr_config_file)
    jq '.alpha = "None"' <<<"$jsonStr" > $bpr_config_file
    jsonStr=$(cat $bpr_config_file)
    jq '.gamma = "None"' <<<"$jsonStr" > $bpr_config_file
    python bpr.py $bpr_config_file
    # boratto sampling
    # rm -rf ./data/$datasets_path_name[$dataset_name]/preprocessed_data/baselineboratto
    # rm -rf ./data/$datasets_path_name[$dataset_name]/sparse_matrices/baselineboratto
    jsonStr=$(cat $bpr_config_file)
    jq '.model_type = "model_types.BASELINE"' <<<"$jsonStr" > $bpr_config_file
    jsonStr=$(cat $bpr_config_file)
    jq '.data_loader_type = "boratto"' <<<"$jsonStr" > $bpr_config_file
    python bpr.py $bpr_config_file
    # boratto reweighting
    jsonStr=$(cat $bpr_config_file)
    jq '.data_loader_type = ""' <<<"$jsonStr" > $bpr_config_file
    for boratto_reg in $(seq 0.1 0.1 1.5); do
      boratto_reg="${boratto_reg//,/.}"
      jsonStr=$(cat $bpr_config_file)
      jq '.model_type = "model_types.BASELINE"' <<<"$jsonStr" > $bpr_config_file
      jsonStr=$(cat $bpr_config_file)
      jq '.reg_weight = "'$boratto_reg'"' <<<"$jsonStr" > $bpr_config_file
      jsonStr=$(cat $bpr_config_file)
      jq '.regularizer = "boratto"' <<<"$jsonStr" > $bpr_config_file
      python bpr.py $bpr_config_file
    done
    # boratto sampling + reweighting
    jsonStr=$(cat $bpr_config_file)
    jq '.data_loader_type = "boratto"' <<<"$jsonStr" > $bpr_config_file
    for boratto_reg in $(seq 0.1 0.1 1.5); do
      boratto_reg="${boratto_reg//,/.}"
      jsonStr=$(cat $bpr_config_file)
      jq '.model_type = "model_types.BASELINE"' <<<"$jsonStr" > $bpr_config_file
      jsonStr=$(cat $bpr_config_file)
      jq '.reg_weight = "'$boratto_reg'"' <<<"$jsonStr" > $bpr_config_file
      python bpr.py $bpr_config_file
    done
    # CAUSAL PD
    jsonStr=$(cat $bpr_config_file)
    jq '.data_loader_type = ""' <<<"$jsonStr" > $bpr_config_file
    for pd_reg in $(seq 0.02 0.02 0.25); do
      pd_reg="${pd_reg//,/.}"
      jsonStr=$(cat $bpr_config_file)
      jq '.model_type = "model_types.BASELINE"' <<<"$jsonStr" > $bpr_config_file
      jsonStr=$(cat $bpr_config_file)
      jq '.reg_weight = "'$pd_reg'"' <<<"$jsonStr" > $bpr_config_file
      jsonStr=$(cat $bpr_config_file)
      jq '.regularizer = "PD"' <<<"$jsonStr" > $bpr_config_file
      python bpr.py $bpr_config_file
    done
  done
done