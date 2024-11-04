"""
    This module creates the model and tokenizer from the HuggingFace CodeGen-Mono 350M model
    https://huggingface.co/Salesforce/codegen-350M-mono.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import tyro


def main(
        model_name: str,
        model_save_name: str,
        tokenizer_save_path_name: str,
):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto").to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    torch.save(model, model_save_name)
    tokenizer.save_pretrained(tokenizer_save_path_name)
    print(f"Model and tokenizer saved to {model_save_name} and {tokenizer_save_path_name}")


if __name__ == "__main__":
    tyro.cli(main)
