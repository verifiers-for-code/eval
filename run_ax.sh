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

# Define the config file path
CONFIG_FILE="ft_yamls/${ADAPTER_NAME}.yml"

# Define the LORA model directory
LORA_MODEL_DIR="adapters/${ADAPTER_NAME}"

# Run the new merge command
python3 -m axolotl.cli.merge_lora ${CONFIG_FILE} --lora_model_dir="${LORA_MODEL_DIR}"

# The merged model will be in ${LORA_MODEL_DIR}/merged
MERGED_MODEL_PATH="${LORA_MODEL_DIR}/merged"

# Run the remaining commands with the provided inputs
python3 rule_inference_json.py --name ${ADAPTER_NAME} --model_path ${MERGED_MODEL_PATH} --temp 0 --tensor_parallel 1

python3 clean_rules_json.py --input_file jsons/${ADAPTER_NAME}_rules.json

python3 gen_eval_json.py --json_file jsons/cleaned-${ADAPTER_NAME}_rules.json --model ${MODEL_NAME} --gpu 2 --subdir ${SUBDIR_NAME} --key cleaned-output

echo "All commands executed successfully."
