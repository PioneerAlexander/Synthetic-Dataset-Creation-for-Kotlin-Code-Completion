"""
Utils for model finetuning pipeline
"""
from typing import List

def print_trainable_parameters(model):
    """
        Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def tokenize(code: List[str], tokenizer, max_length: int = 256):
    complete_code = [item.split("SEP")[0] for item in code]
    prompts = [item.split("SEP")[1] for item in code]
    tokenized_code = tokenizer(
        complete_code,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        add_special_tokens=True,
    )

    tokenized_code["labels"] = tokenized_code["input_ids"].clone()
    
    for idx, prompt in enumerate(prompts):
        if prompt != "-1":
            masked_tokens = tokenizer(
                prompt, 
                return_tensors="pt"
            ).input_ids[0]

            tokenized_code["labels"][idx, :len(masked_tokens)] = -100       
    
    return tokenized_code

def collate_fn(batch, tokenizer, max_length):
    code = [item for item in batch]

    return tokenize(code, tokenizer, max_length)

