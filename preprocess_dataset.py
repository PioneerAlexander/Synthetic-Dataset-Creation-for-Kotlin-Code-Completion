"""
    This module is aimed to preprocess and save training and validation datasets
"""
from typing import List, Dict
import json
import os

from sklearn.model_selection import train_test_split

import tyro

def save_dataset(dataset: List[Dict[str, str]], dataset_save_path: str):
    """
    Save the dataset in the format [{"code": ..., "prompt": ...}, {"code": ..., "prompt": ...}]
    Args:
        dataset (List[Dict[str, str]]): code samples to save
        dataset_save_path (str): path where to save
    """

    with open(dataset_save_path, "w", encoding="utf-8") as f:
        for code in dataset:
            f.write(json.dumps(code, indent=None))
            f.write("\n")


def remove_comment_lines(text: str) -> str:
    """
    Removes lines that start with '//' from the given text.
    
    Args:
        text (str): The input text containing lines with comments.
    Returns:
        str: The text with comment lines removed.
    """
    lines = text.splitlines()
    filtered_lines = [line for line in lines if not line.strip().startswith("//")]
    return "\n".join(filtered_lines)


def preprocess_problem_solution_pairs(code_task: Dict["str", "str"]) -> Dict["str", "str"]:
    """
    Create code tasks corpus on which we plan to finetune our model

    Args:
        code_task (Dict[str, str]): problem and solution of the task
    Returns:
        (Dict[str, str]) full task code + prompt (problem + solution, problem)
    """
    problem = code_task["problem"]
    solution = code_task["solution"]

    solution = solution.removeprefix("```kotlin")
    solution = solution.removeprefix("```")

    solution = solution.removesuffix("```")

    solution = remove_comment_lines(solution)

    if not solution.endswith("}"):
        solution += "\n}"
    

    if solution.find("fun") != -1:
        # gpt is likely repeating a problem signature
        return {"code": solution, "prompt": "-1"} # no prompt masking

    else:
        problem = problem.removesuffix("}")
        return {"code": problem + solution, "prompt": problem}

def main(
    generated_synthetic_dataset_filename: str,
    dataset_save_path: str
):

    with open(generated_synthetic_dataset_filename, "r", encoding="utf-8") as f:
        synthetic_dataset = json.load(f)

    train = [
        preprocess_problem_solution_pairs(code_task) for code_task in synthetic_dataset
    ]

    train, val = train_test_split(train, test_size=0.1, random_state=42)

    save_dataset(train, os.path.join(dataset_save_path, "train.jsonl"))
    save_dataset(val, os.path.join(dataset_save_path, "val.jsonl"))


if __name__ == "__main__":
    tyro.cli(main) 

