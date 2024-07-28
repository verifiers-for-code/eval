# Format of .sh
1. Using unsloth merge
```bash
./run.sh different_adapter different_subdir "different/model/path"
```
Example:
```bash
./run.sh phi_512 phi_512_planner "meta-llama/Meta-Llama-3B-Instruct"
```
Here, it runs the model in `meta-llama/Meta-Llama-3B-Instruct` (codegen) with `phi_512` adapter (planner) and `phi_512_planner` subdir (the name you will see on leaderboard).

2. Using axolotl merge
```bash
./run_ax.sh phi_512 phi_512_planner "meta-llama/Meta-Llama-3B-Instruct"
```
Here, it runs the model in `meta-llama/Meta-Llama-3B-Instruct` (codegen) with `phi_512` adapter (planner) and `phi_512_planner` subdir (the name you will see on leaderboard).
Where:
```bash
# Define the config file path
CONFIG_FILE="ft_yamls/${ADAPTER_NAME}.yml"

# Define the LORA model directory
LORA_MODEL_DIR="adapters/${ADAPTER_NAME}"

# Run the new merge command
python3 -m axolotl.cli.merge_lora ${CONFIG_FILE} --lora_model_dir="${LORA_MODEL_DIR}"

# The merged model will be in ${LORA_MODEL_DIR}/merged
MERGED_MODEL_PATH="${LORA_MODEL_DIR}/merged"
```

# Visualize Only Mode
```bash
python visualize_scores.py --model "Meta-Llama-3B-Instruct"
```
This will showcase all the scores of the model in the `model` directory. Ex: if phi is codegen, then ```--model Phi-3-mini-4k-instruct```