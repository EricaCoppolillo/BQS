# pick dataset(s) of interest - MOVIELENS_1M MOVIELENS_20M CITEULIKE PINTEREST EPINIONS NETFLIX YAHOO COAT AMAZON_GGF
declare -a datasets=("MOVIELENS_1M" "YAHOO" "PINTEREST" "CITEULIKE")

cuda="2"
bpr_config_file="simgcl_config.json"
num_epochs=20

# moving in the parent directory
cd ../
# (12121995 190291 230782 81163 100362)
declare -a seeds=()
declare -a seeds_tmp=(12121995 190291 230782 81163 100362)

for seed in "${seeds_tmp[@]}"; do
  for dataset_name in "${datasets[@]}"; do
    # baseline
    jsonStr=$(cat $bpr_config_file)
    jq '.seed = '$seed <<<"$jsonStr" > $bpr_config_file
    jsonStr=$(cat $bpr_config_file)
    jq '.dataset_name = "datasets.'$dataset_name'"' <<<"$jsonStr" > $bpr_config_file
    jsonStr=$(cat $bpr_config_file)
    jq '.model_type = "model_types.LOW"' <<<"$jsonStr" > $bpr_config_file
    jsonStr=$(cat $bpr_config_file)
    jq '.regularizer = ""' <<<"$jsonStr" > $bpr_config_file
    jsonStr=$(cat $bpr_config_file)
    jq '.n_epochs = '$num_epochs <<<"$jsonStr" > $bpr_config_file
    jsonStr=$(cat $bpr_config_file)
    jq '.data_loader_type = ""' <<<"$jsonStr" > $bpr_config_file
    jsonStr=$(cat $bpr_config_file)
    jq '.CUDA_VISIBLE_DEVICES = "'$cuda'"' <<<"$jsonStr" > $bpr_config_file
    jsonStr=$(cat $bpr_config_file)
    jq '.dir_name = "low_'$seed'"' <<<"$jsonStr" > $bpr_config_file
    jsonStr=$(cat $bpr_config_file)
    lambda_val=0.01
    jq '.lambda_val = '$lambda_val <<<"$jsonStr" > $bpr_config_file
    python simGCL.py $bpr_config_file
  done
done

for seed in "${seeds[@]}"; do
  for dataset_name in "${datasets[@]}"; do
      # baseline
      jsonStr=$(cat $bpr_config_file)
      jq '.seed = '$seed <<<"$jsonStr" > $bpr_config_file
      jsonStr=$(cat $bpr_config_file)
      jq '.dataset_name = "datasets.'$dataset_name'"' <<<"$jsonStr" > $bpr_config_file
      jsonStr=$(cat $bpr_config_file)
      jq '.model_type = "model_types.BASELINE"' <<<"$jsonStr" > $bpr_config_file
      jsonStr=$(cat $bpr_config_file)
      jq '.regularizer = ""' <<<"$jsonStr" > $bpr_config_file
      jsonStr=$(cat $bpr_config_file)
      jq '.n_epochs = '$num_epochs <<<"$jsonStr" > $bpr_config_file
      jsonStr=$(cat $bpr_config_file)
      jq '.data_loader_type = ""' <<<"$jsonStr" > $bpr_config_file
      jsonStr=$(cat $bpr_config_file)
      jq '.CUDA_VISIBLE_DEVICES = "'$cuda'"' <<<"$jsonStr" > $bpr_config_file
      jsonStr=$(cat $bpr_config_file)
      jq '.dir_name = "baseline_'$seed'"' <<<"$jsonStr" > $bpr_config_file
      jsonStr=$(cat $bpr_config_file)
      lambda_val=0.01
      jq '.lambda_val = '$lambda_val <<<"$jsonStr" > $bpr_config_file
      python simGCL.py $bpr_config_file

      # oversampling
      jsonStr=$(cat $bpr_config_file)
      jq '.model_type = "model_types.OVERSAMPLING"' <<<"$jsonStr" > $bpr_config_file
      jsonStr=$(cat $bpr_config_file)
      jq '.dir_name = "oversampling_'$seed'"' <<<"$jsonStr" > $bpr_config_file
      python simGCL.py $bpr_config_file

      # uniform oversampling
      jsonStr=$(cat $bpr_config_file)
      jq '.model_type = "model_types.U_SAMPLING"' <<<"$jsonStr" > $bpr_config_file
      jsonStr=$(cat $bpr_config_file)
      jq '.dir_name = "usampling_'$seed'"' <<<"$jsonStr" > $bpr_config_file
      python simGCL.py $bpr_config_file

      # Competitors
      # jannach
      jannach_weight="2.25"
      jsonStr=$(cat $bpr_config_file)
      jq '.model_type = "model_types.BASELINE"' <<<"$jsonStr" > $bpr_config_file
      jsonStr=$(cat $bpr_config_file)
      jq '.jannach_width = "'$jannach_weight'"' <<<"$jsonStr" > $bpr_config_file
      jsonStr=$(cat $bpr_config_file)
      jq '.regularizer = ""' <<<"$jsonStr" > $bpr_config_file
      jsonStr=$(cat $bpr_config_file)
      jq '.data_loader_type = "jannach"' <<<"$jsonStr" > $bpr_config_file
      jsonStr=$(cat $bpr_config_file)
      jq '.dir_name = "jannach_'$seed'_'$jannach_weight'"' <<<"$jsonStr" > $bpr_config_file
      python simGCL.py $bpr_config_file

      # ips
      jsonStr=$(cat $bpr_config_file)
      jq '.data_loader_type = ""' <<<"$jsonStr" > $bpr_config_file
      jsonStr=$(cat $bpr_config_file)
      jq '.model_type = "model_types.REWEIGHTING"' <<<"$jsonStr" > $bpr_config_file
      jsonStr=$(cat $bpr_config_file)
      jq '.alpha = "None"' <<<"$jsonStr" > $bpr_config_file
      jsonStr=$(cat $bpr_config_file)
      jq '.gamma = "None"' <<<"$jsonStr" > $bpr_config_file
      jsonStr=$(cat $bpr_config_file)
      jq '.dir_name = "ips_'$seed'"' <<<"$jsonStr" > $bpr_config_file
      python simGCL.py $bpr_config_file

      # boratto reweighting
      boratto_reg="0.7"
      jsonStr=$(cat $bpr_config_file)
      jq '.model_type = "model_types.BASELINE"' <<<"$jsonStr" > $bpr_config_file
      jsonStr=$(cat $bpr_config_file)
      jq '.reg_weight = "'$boratto_reg'"' <<<"$jsonStr" > $bpr_config_file
      jsonStr=$(cat $bpr_config_file)
      jq '.regularizer = "boratto"' <<<"$jsonStr" > $bpr_config_file
      jsonStr=$(cat $bpr_config_file)
      jq '.dir_name = "borattoreweighting_'$seed'_'$boratto_reg'"' <<<"$jsonStr" > $bpr_config_file
      python simGCL.py $bpr_config_file

      # CAUSAL PD
      jsonStr=$(cat $bpr_config_file)
      jq '.data_loader_type = ""' <<<"$jsonStr" > $bpr_config_file
      pd_reg="0.12"
      jsonStr=$(cat $bpr_config_file)
      jq '.model_type = "model_types.BASELINE"' <<<"$jsonStr" > $bpr_config_file
      jsonStr=$(cat $bpr_config_file)
      jq '.reg_weight = "'$pd_reg'"' <<<"$jsonStr" > $bpr_config_file
      jsonStr=$(cat $bpr_config_file)
      jq '.regularizer = "PD"' <<<"$jsonStr" > $bpr_config_file
      jsonStr=$(cat $bpr_config_file)
      jq '.dir_name = "pd_'$seed'_'$pd_reg'"' <<<"$jsonStr" > $bpr_config_file
      python3 simGCL.py $bpr_config_file

  done
done