"""
This module translate python code exercises dataset to Kotlin using ChatGPT API.
"""

import json

from datasets import load_dataset
from openai import OpenAI 
from tqdm import tqdm

import tyro
import heapq

def chat_gpt(prompt, client):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()


def main(
    dataset_save_name: str,
    dataset_size: int,
):
    
    client = OpenAI(
        api_key=open(".api_key").read().strip(),
    )
    ds = load_dataset("jinaai/code_exercises")

    tasks_with_shortest_solution = heapq.nsmallest(dataset_size, ds["train"], key=lambda x: x['solution'])

    for task in tqdm(tasks_with_shortest_solution):
        problem_translation = chat_gpt(f"Translate problem definition from Python to Kotlin. Don't finish the code.: {item['problem']}")

        solution_translation = chat_gpt(f"Translate solution from Python to Kotlin. Don't comment the process.: {item['solution']}")
        with open(dataset_save_name, "r") as file:
            synthetic_dataset = json.load(file)
        synthetic_dataset.append({
            "problem": problem_translation,
            "solution": solution_translation
        })
        with open(dataset_save_name, "w") as file:
            json.dump(synthetic_dataset, file, indent=4)

if __name__ == "__main__":
    tyro.cli(main)