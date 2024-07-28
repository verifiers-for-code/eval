#!/bin/bash

# Check if all required arguments are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <adapter_name> <subdir_name> <model_name>"
    exit 1
fi

# Assign input arguments to variables
ADAPTER_NAME=$1
SUBDIR_NAME=$2
MODEL_NAME=$3

# Run the commands with the provided inputs
python3 merge_adapters.py --adapter adapters/${ADAPTER_NAME} --out merged/merged_${ADAPTER_NAME} --max_seq_length 4096

python3 rule_inference_json.py --name ${ADAPTER_NAME} --model_path merged/merged_${ADAPTER_NAME} --temp 0 --tensor_parallel 1

python3 clean_rules_json.py --input_file jsons/${ADAPTER_NAME}_rules.json

python3 gen_eval_json.py --json_file jsons/cleaned-${ADAPTER_NAME}_rules.json --model ${MODEL_NAME} --gpu 2 --subdir ${SUBDIR_NAME} --key cleaned-output

echo "All commands executed successfully."
