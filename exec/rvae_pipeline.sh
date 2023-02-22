# pick dataset(s) of interest - MOVIELENS_1M CITEULIKE PINTEREST YAHOO AMAZON_GGF
declare -a datasets=("PINTEREST" "AMAZON_GGF" "YAHOO" "PINTEREST" "CITEULIKE" "MOVIELENS_1M")

cuda="3"
# moving in the parent directory
cd ../
declare -a seeds=(12121995 230782 190291 81163 100362)

for seed in "${seeds[@]}"; do

  for dataset_name in "${datasets[@]}"; do
    echo $dataset_name
    # baseline
    jsonStr=$(cat rvae_config.json)
    jq '.seed = '$seed <<<"$jsonStr" > rvae_config.json
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
    jsonStr=$(cat rvae_config.json)
    jq '.dir_name = "baseline_'$seed'"' <<<"$jsonStr" > rvae_config.json
    python3 rvae.py

    # oversampling
    jsonStr=$(cat rvae_config.json)
    jq '.model_type = "model_types.OVERSAMPLING"' <<<"$jsonStr" > rvae_config.json
    jsonStr=$(cat rvae_config.json)
    jq '.dir_name = "oversampling_'$seed'"' <<<"$jsonStr" > rvae_config.json
    python3 rvae.py

#   Competitors
#   jannach
    jannach_weight="2.25"

    jsonStr=$(cat rvae_config.json)
    jq '.model_type = "model_types.BASELINE"' <<<"$jsonStr" > rvae_config.json
    jsonStr=$(cat rvae_config.json)
    jq '.jannach_width = "'$jannach_weight'"' <<<"$jsonStr" > rvae_config.json
    jsonStr=$(cat rvae_config.json)
    jq '.regularizer = ""' <<<"$jsonStr" > rvae_config.json
    jsonStr=$(cat rvae_config.json)
    jq '.data_loader_type = "jannach"' <<<"$jsonStr" > rvae_config.json
    jsonStr=$(cat rvae_config.json)
    jq '.dir_name = "jannach_'$seed'_'$jannach_weight'"' <<<"$jsonStr" > rvae_config.json
    python3 rvae.py

    # ips
    jsonStr=$(cat rvae_config.json)
    jq '.data_loader_type = ""' <<<"$jsonStr" > rvae_config.json
    jsonStr=$(cat rvae_config.json)
    jq '.model_type = "model_types.REWEIGHTING"' <<<"$jsonStr" > rvae_config.json
    jsonStr=$(cat rvae_config.json)
    jq '.alpha = "None"' <<<"$jsonStr" > rvae_config.json
    jsonStr=$(cat rvae_config.json)
    jq '.gamma = "None"' <<<"$jsonStr" > rvae_config.json
    jsonStr=$(cat rvae_config.json)
    jq '.dir_name = "ips_'$seed'"' <<<"$jsonStr" > rvae_config.json
    python3 rvae.py

    boratto_reg="0.7"
    jsonStr=$(cat rvae_config.json)
    jq '.model_type = "model_types.BASELINE"' <<<"$jsonStr" > rvae_config.json
    jsonStr=$(cat rvae_config.json)
    jq '.reg_weight = "'$boratto_reg'"' <<<"$jsonStr" > rvae_config.json
    jsonStr=$(cat rvae_config.json)
    jq '.regularizer = "boratto"' <<<"$jsonStr" > rvae_config.json
    jsonStr=$(cat rvae_config.json)
    jq '.dir_name = "borattoreweighting_'$seed'_'$boratto_reg'"' <<<"$jsonStr" > rvae_config.json
    python3 rvae.py

     # CAUSAL PD
    pd_reg="0.12"
    jsonStr=$(cat rvae_config.json)
    jq '.model_type = "model_types.BASELINE"' <<<"$jsonStr" > rvae_config.json
    jsonStr=$(cat rvae_config.json)
    jq '.reg_weight = "'$pd_reg'"' <<<"$jsonStr" > rvae_config.json
    jsonStr=$(cat rvae_config.json)
    jq '.regularizer = "PD"' <<<"$jsonStr" > rvae_config.json
    jsonStr=$(cat rvae_config.json)
    jq '.dir_name = "pd_'$seed'_'$pd_reg'"' <<<"$jsonStr" > rvae_config.json
    python3 rvae.py

  done
done

# run average_runs on competitors
