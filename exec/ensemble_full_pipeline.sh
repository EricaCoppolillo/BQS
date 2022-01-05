# pick dataset(s) of interest - MOVIELENS_1M MOVIELENS_20M CITEULIKE PINTEREST EPINIONS NETFLIX YAHOO COAT AMAZON_GGF
declare -a datasets=("AMAZON_GGF")
declare -A datasets_path_name=( ["MOVIELENS_1M"]="ml-1m" ["MOVIELENS_20M"]="ml-20m" ["CITEULIKE"]="citeulike-a" ["PINTEREST"]="pinterest" ["EPINIONS"]="epinions" ["NETFLIX"]="netflix", ["YAHOO"]="yahoo-r3", ["COAT"]="coat",  ["AMAZON_GGF"]="amzn-ggf")
cuda="2"
# moving in the parent directory
cd ../

for seed in "${seeds[@]}"; do
  # iterating across datasets
  for dataset_name in "${datasets[@]}"; do
    # loading default config file - TODO: Add capability to use custom files
    cat default_rvae_configs.json | python -c "import sys, json; from util import clean_json_string; x=json.load(sys.stdin); x=x[sys.argv[1]]; x['baseline_flag']=str("True"); print(clean_json_string(str(x)))" $dataset_name > rvae_config.json
    # running the baseline model training
    python rvae.py
    # loading default config file - switching baseline flag
    cat default_rvae_configs.json | python -c "import sys, json; from util import clean_json_string; x=json.load(sys.stdin); x=x[sys.argv[1]]; x['baseline_flag']=str("False"); print(clean_json_string(str(x)))" $dataset_name > rvae_config.json
    # running low popular model training
    python rvae.py
    # loading default config file - TODO: Add capability to use custom files
    cat default_ensemble_configs.json | python -c "import sys, json; from util import clean_json_string; x=json.load(sys.stdin); x=x[sys.argv[1]]; print(clean_json_string(str(x)))" $dataset_name > ensemble_config.json
    # running the ensemble
    python ensemble.py
  done
done