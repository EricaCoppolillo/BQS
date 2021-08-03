# pick dataset(s) of interest - MOVIELENS_1M MOVIELENS_20M CITEULIKE PINTEREST EPINIONS NETFLIX
cuda="0"
# moving in the parent directory
cd $(dirname $0)/../

DATASETS="CITEULIKE 0.2 0.05 0.7
MOVIELENS_1M 0.2 0.05 0.7
PINTEREST 0.5 0.05 0.8
EPINIONS 0.3 0.05 0.45
MOVIELENS_20M 2.0 0.1 3.0
NETFLIX 0.25 0.05 0.4
"

DATASETS="NETFLIX 0.25 0.05 0.4
"

# iterating across datasets
#for dataset_name in "${datasets[@]}"; do

echo -ne "$DATASETS" | while read line; do
  dataset_name=$(echo -ne "$line" | awk '{print $1}')
  params=$(echo -ne "$line" | awk '{print $2, $3, $4}')

  echo "$dataset_name -- $params"
  sleep 5s
  for i in $(seq $params); do
    new_i="${i//,/.}"
    # loading default config file - TODO: Add capability to use custom files
    cat default_ensemble_configs.json | python -c "import sys, json; from util import clean_json_string; x=json.load(sys.stdin); x=x[sys.argv[1]]; x['CUDA_VISIBLE_DEVICES']=sys.argv[2]; x['ensemble_weight']=float(sys.argv[3]); print(clean_json_string(str(x)))" $dataset_name $cuda $new_i > ensemble_config.json
    echo "Currently analyzing: "$dataset_name
    # running the baseline model training
    python ensemble.py
    sleep 1m
    sleep 1s
  done
done