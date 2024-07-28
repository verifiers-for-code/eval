import argparse
import json
import time
from vllm import LLM, SamplingParams

SYSTEM_PROMPT = "You are given the start of a function for a Python program. Your job is to produce a detailed plan. First, analyze and think, by using <thinking> XML tags, and then produce a plan, in <plan> tags."

def create_prompts(prompts, tokenizer):
    formatted_prompts = []
    for prompt in prompts:
        x = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            tokenize=False,
            add_generation_prompt=True
        )
        formatted_prompts.append(x)
    return formatted_prompts

def main():
    parser = argparse.ArgumentParser(description="Process prompts using vLLM and save results.")
    parser.add_argument("--name", type=str, required=True, help="Name for the model and output JSON file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the local model directory")
    parser.add_argument("--temp", type=float, required=True, help="Temperature for inference")
    parser.add_argument("--sleep", type=int, default=0, help="Optional delay in seconds before starting the main process")
    parser.add_argument("--tensor_parallel", type=int, default=1, help="Tensor parallel size for vLLM")
    args = parser.parse_args()

    MODEL_NAME = args.name
    JSON_INPUT_FILENAME = "jsons/eval_prompts.json"
    JSON_OUTPUT_FILENAME = f"jsons/{MODEL_NAME}_rules.json"

    # Initialize vLLM and tokenizer
    llm = LLM(model=args.model_path, tensor_parallel_size=args.tensor_parallel, gpu_memory_utilization=0.95)
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(temperature=args.temp, max_tokens=2048)

    if args.sleep > 0:
        print(f"Sleeping for {args.sleep} seconds before starting...")
        time.sleep(args.sleep)

    # Load prompts from JSON file
    with open(JSON_INPUT_FILENAME, 'r') as f:
        eval_prompts = json.load(f)

    prompts = [item['prompt'] for item in eval_prompts]
    formatted_prompts = create_prompts(prompts, tokenizer)

    print(f"Processing {len(formatted_prompts)} prompts in a single batch with tensor parallelism...")
    outputs = llm.generate(formatted_prompts, sampling_params)

    results = [
        {'prompt': prompt, 'output': output.outputs[0].text}
        for prompt, output in zip(prompts, outputs)
    ]

    # Save results to JSON file
    try:
        with open(JSON_OUTPUT_FILENAME, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Successfully saved results to {JSON_OUTPUT_FILENAME}")
    except Exception as e:
        print(f"Failed to save results to disk: {e}")

if __name__ == "__main__":
    main()
