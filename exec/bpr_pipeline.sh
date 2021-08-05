# pick dataset(s) of interest - MOVIELENS_1M MOVIELENS_20M CITEULIKE PINTEREST EPINIONS NETFLIX
#declare -a datasets=("MOVIELENS_1M" "CITEULIKE" "PINTEREST" "EPINIONS" "MOVIELENS_20M" "NETFLIX")
declare -a datasets=("EPINIONS" "NETFLIX")
declare -A datasets_path_name=( ["MOVIELENS_1M"]="ml-1m" ["MOVIELENS_20M"]="ml-20m" ["CITEULIKE"]="citeulike-a" ["PINTEREST"]="pinterest" ["EPINIONS"]="epinions" ["NETFLIX"]="netflix")
cuda="0"
# moving in the parent directory
cd $(dirname $0)/../

rm -f last_log_bprpipeline.txt

# iterating across datasets
for dataset_name in "${datasets[@]}"; do
  # loading default config file - TODO: Add capability to use custom files
  echo "Currently analyzing: "$dataset_name
  cat default_bpr_configs.json | python -c "import sys, json; from util import clean_json_string; x=json.load(sys.stdin); x=x[sys.argv[1]]; x['CUDA_VISIBLE_DEVICES']=sys.argv[2]; print(clean_json_string(str(x)))" $dataset_name $cuda > bpr_config.json
  # running the LOW model training
  python bpr.py 2>&1 | tee -a last_log_bprpipeline.txt
done
