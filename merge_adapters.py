import argparse
from unsloth import FastLanguageModel

def main():
    parser = argparse.ArgumentParser(description="FastLanguageModel script with command-line arguments")
    parser.add_argument("--adapter", required=True, help="Model name (adapter)")
    parser.add_argument("--out", required=True, help="Output file name for merged model")
    parser.add_argument("--max_seq_length", type=int, default=4096, help="Max sequence length (default: 4096)")
    parser.add_argument("--load_in_4bit", action="store_true", help="Use 4-bit quantization")
    args = parser.parse_args()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.adapter,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
    )

    # Load the LoRA adapter
    # Uncomment and modify the following lines if you need to load a LoRA adapter
    # model = FastLanguageModel.get_peft_model(
    #     model,
    #     "path/to/your/adapter_model.safetensors",  # Replace with the path to your adapter files
    # )

    model.save_pretrained_merged(args.out, tokenizer, save_method="merged_16bit")

if __name__ == "__main__":
    main()
