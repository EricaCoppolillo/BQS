# pick dataset(s) of interest - MOVIELENS_1M MOVIELENS_20M CITEULIKE PINTEREST EPINIONS NETFLIX
declare -a datasets=("MOVIELENS_1M")
declare -A datasets_path_name=( ["MOVIELENS_1M"]="ml-1m" ["MOVIELENS_20M"]="ml-20m" ["CITEULIKE"]="citeulike-a" ["PINTEREST"]="pinterest" ["EPINIONS"]="epinions" ["NETFLIX"]="netflix")
# moving in the parent directory
cd ../

# iterating across datasets
for dataset_name in "${datasets[@]}"; do
  # loading default config file - TODO: Add capability to use custom files
  cat default_rvae_configs.json | python -c "import sys, json; from util import clean_json_string; x=json.load(sys.stdin); x=x[sys.argv[1]]; x['baseline_flag']=str("True"); print(clean_json_string(str(x)))" $dataset_name > rvae_config.json
  # running the baseline model training
  baseline_run_dir=$(python rvae.py "${datasets_path_name[$dataset_name]}")
  baseline_run_dir=$(echo $baseline_run_dir | rev | cut -d " " -f1 | rev)
  # loading default config file - switching baseline flag
  cat default_rvae_configs.json | python -c "import sys, json; from util import clean_json_string; x=json.load(sys.stdin); x=x[sys.argv[1]]; x['baseline_flag']=str("False"); print(clean_json_string(str(x)))" $dataset_name > rvae_config.json
  # running low popular model training
  low_pop_run_dir=$(python rvae.py "${datasets_path_name[$dataset_name]}")
  low_pop_run_dir=$(echo $low_pop_run_dir | rev | cut -d " " -f1 | rev)
  # moving the two running directory in the main folder in order to be processable by the ensemble script, before doing that you need to delete the current folders
  rm -rf ./data/"${datasets_path_name[$dataset_name]}"/baseline
  cp -r $baseline_run_dir ./data/"${datasets_path_name[$dataset_name]}"/baseline
  rm -rf ./data/"${datasets_path_name[$dataset_name]}"/popularity_low
  cp -r $low_pop_run_dir ./data/"${datasets_path_name[$dataset_name]}"/popularity_low
  # loading default config file - TODO: Add capability to use custom files
  cat default_ensemble_configs.json | python -c "import sys, json; from util import clean_json_string; x=json.load(sys.stdin); x=x[sys.argv[1]]; print(clean_json_string(str(x)))" $dataset_name > ensemble_config.json
  # executing the ensemble
  python ensemble.py "${datasets_path_name[$dataset_name]}"
done