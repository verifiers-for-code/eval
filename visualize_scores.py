import os
import subprocess
import argparse
from tabulate import tabulate
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

def run_evaluation(output_dir, subdir):
    command = f"evalplus.evaluate --dataset humaneval --samples {os.path.join(output_dir, subdir)}"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout

def parse_scores(output):
    lines = output.split('\n')
    humaneval_score = None
    humaneval_plus_score = None
    for line in lines:
        if "humaneval (base tests)" in line:
            humaneval_score = float(lines[lines.index(line) + 1].split(':')[1].strip()) * 100
        elif "humaneval+ (base + extra tests)" in line:
            humaneval_plus_score = float(lines[lines.index(line) + 1].split(':')[1].strip()) * 100
    return humaneval_score, humaneval_plus_score

def color_row(subdir, humaneval_score, humaneval_plus_score):
    if subdir == "gold_plan":
        color = Fore.GREEN
    elif subdir == "none":
        color = Fore.BLUE
    else:
        color = ""
    
    return [
        f"{color}{subdir}{Style.RESET_ALL}",
        f"{color}{humaneval_score:.3f}{Style.RESET_ALL}",
        f"{color}{humaneval_plus_score:.3f}{Style.RESET_ALL}"
    ]

def main(model, sort_eval_plus):
    output_dir = f"{model}-output"
    results = []
    for subdir in os.listdir(output_dir):
        subdir_path = os.path.join(output_dir, subdir)
        if os.path.isdir(subdir_path):
            eval_output = run_evaluation(output_dir, subdir)
            humaneval_score, humaneval_plus_score = parse_scores(eval_output)
            results.append((subdir, humaneval_score, humaneval_plus_score))
    
    # Sort results based on the selected score
    if sort_eval_plus:
        results.sort(key=lambda x: x[2], reverse=True)  # Sort by HumanEval+ score
    else:
        results.sort(key=lambda x: x[1], reverse=True)  # Sort by HumanEval score
    
    # Apply coloring after sorting
    colored_results = [color_row(*result) for result in results]
    
    # Change the order of columns based on sorting method
    if sort_eval_plus:
        headers = ["Subdir", "HumanEval+ (%)", "HumanEval (%)"]
        colored_results = [[row[0], row[2], row[1]] for row in colored_results]
    else:
        headers = ["Subdir", "HumanEval (%)", "HumanEval+ (%)"]
    
    print(tabulate(colored_results, headers=headers, tablefmt="grid"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate and compare HumanEval results.")
    parser.add_argument("--model", required=True, help="Model name (e.g., Meta-llama-3-8b-instruct)")
    parser.add_argument("--sort-eval-plus", action="store_true", help="Sort by HumanEval+ score when specified")
    args = parser.parse_args()

    main(args.model, args.sort_eval_plus)
