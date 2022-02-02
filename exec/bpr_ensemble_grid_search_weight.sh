# pick dataset(s) of interest - MOVIELENS_1M MOVIELENS_20M CITEULIKE PINTEREST EPINIONS NETFLIX YAHOO COAT AMAZON_GGF
declare -a datasets=("AMAZON_GGF")
declare -A datasets_path_name=( ["MOVIELENS_1M"]="ml-1m" ["MOVIELENS_20M"]="ml-20m" ["CITEULIKE"]="citeulike-a" ["PINTEREST"]="pinterest" ["EPINIONS"]="epinions" ["NETFLIX"]="netflix", ["YAHOO"]="yahoo-r3", ["COAT"]="coat",  ["AMAZON_GGF"]="amzn-ggf")
declare -A dataset2alpha=( ["MOVIELENS_1M"]="0.01" ["MOVIELENS_20M"]="0.0025" ["CITEULIKE"]="0.05" ["PINTEREST"]="0.05" ["EPINIONS"]="0.1" ["NETFLIX"]="0.0005", ["YAHOO"]="0.02", ["COAT"]="0.5",  ["AMAZON_GGF"]="0.01")
declare -A dataset2gamma=( ["MOVIELENS_1M"]="53.33" ["MOVIELENS_20M"]="10" ["CITEULIKE"]="6.67" ["PINTEREST"]="10" ["EPINIONS"]="6.67" ["NETFLIX"]="10", ["YAHOO"]="10", ["COAT"]="7.0",  ["AMAZON_GGF"]="23.33")
declare -A dataset2latent_dim=( ["MOVIELENS_1M"]="64" ["MOVIELENS_20M"]="10" ["CITEULIKE"]="128" ["PINTEREST"]="128" ["EPINIONS"]="6.67" ["NETFLIX"]="10", ["YAHOO"]="32", ["COAT"]="7.0",  ["AMAZON_GGF"]="256")


cuda="1"
ensemble_config_file="ensemble_config.json"

# moving in the parent directory
cd ../
declare -a seeds=(12121995)

for seed in "${seeds[@]}"; do

  for dataset_name in "${datasets[@]}"; do
    # Ensemble
    for ensemble_weight in $(seq 0.05 0.05 1.0); do
      new_ensemble_weight="${ensemble_weight//,/.}"
      jsonStr=$(cat $ensemble_config_file)
      jq '.ensemble_weight = '$new_ensemble_weight <<<"$jsonStr" > $ensemble_config_file
      jsonStr=$(cat $ensemble_config_file)
      jq '.latent_dim = "'${dataset2latent_dim[$dataset_name]}'"' <<<"$jsonStr" > $ensemble_config_file
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
      # running the ensemble model
      python ensemble.py $ensemble_config_file
      sleep 60s
    done
  done
done