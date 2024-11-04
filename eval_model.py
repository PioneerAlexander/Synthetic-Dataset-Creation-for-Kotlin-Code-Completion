"""
    This module contains the run code for the evaluation the loaded model and tokenizer on given
    dataset.
"""
from functools import partial
from typing import Optional
import json

import torch
import tyro
import wandb
from torch.utils.data import DataLoader
from datasets import load_dataset
import jsonlines
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from tqdm import tqdm 
from mxeval.evaluation import evaluate_functional_correctness
from code_postprocess_utils import clean_answer, generate, StoppingCriteriaSub

def main(
        model_load_path: str,
        tokenizer_load_path: str,
        output_file: str,

        /,  # Mark the end of positional arguments

        use_wandb: bool = True,
        wandb_project: Optional[str] = "jb_synthetic_dataset_test_task", # change to your project name if you need
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    assert (not use_wandb) or (use_wandb and wandb_project is not None)

    if use_wandb:
        # Log in to your wandb account with netrc file in your home directory
        # wandb.login(key=get_api_key_from_netrc("api.wandb.ai"))
        wandb.init(
            project=wandb_project,
            reinit=True,
            resume="allow",
            entity="kariakinaleksandr"  # Change to your_username for reproduction
        )

    model = torch.load(model_load_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_load_path)
    dataset = load_dataset("jetbrains/Kotlin_HumanEval")['train']
    problem_dict = {problem['task_id']: problem for problem in dataset}

    output = []
    for key in tqdm(list(problem_dict.keys()), leave=False):
        problem = problem_dict[key]["prompt"]

        answer = generate(problem, tokenizer, model)
        answer = clean_answer(answer)
        output.append({"task_id": key, "completion": answer, "language": "kotlin"})
    
    
    with jsonlines.open(output_file, mode="w") as writer:
        for line in output:
            writer.write(line)

    evaluate_functional_correctness(
        sample_file=output_file,
        k=[1],
        n_workers=16,
        timeout=15,
        problem_file=problem_dict,
    )

    with open(output_file + '_results.jsonl') as fp:
        total = 0
        correct = 0
        for line in fp:
            sample_res = json.loads(line)
            total += 1
            correct += sample_res['passed']

    wandb.log({"pass_rate": correct/total})
    print(f'Pass rate: {correct/total}')
    

if __name__ == "__main__":
    tyro.cli(main)
