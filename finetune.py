"""
    Model fine-tuning pipeline
"""
from typing import Optional

import torch
import tyro
import wandb
from transformers import AutoTokenizer, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig

from src.dataset.FinetuningDataset import FineTuningDataset

from src.model.utils import print_trainable_parameters, collate_fn

import os

from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig

fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
)

accelerator = Accelerator(fsdp_plugin=fsdp_plugin)


def main(
        model_load_file_name: str,
        tokenizer_load_path_name: str,
        dataset_path: str,

        /,  # Mark the end of positional arguments

        max_len: int = 256,
        batch_size: int = 32,
        lr: float = 2.5e-5,
        weight_decay: float = 0.01,
        epochs: int = 1,

        # Lora parameters
        r: int=8,
        lora_alpha: float=16.0,
        lora_dropout: float=0.05, 

        use_wandb: bool = True,
        wandb_project: Optional[str] = "jb_synthetic_dataset_test_task",

        save_last: int = 2,
        save_every: int = 1000,
        eval_every: int = 1000,

        run_name: str = "finetuning-granite",
        final_model_name: str = "finetuned_granite.pth"
):
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

    peft_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "dense",
            "fc1",
            "fc2",
            "lm_head"
        ],
        bias="none",
        lora_dropout=lora_dropout,  
        task_type="CAUSAL_LM",
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = torch.load(model_load_file_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_load_path_name)
    
    model = accelerator.prepare_model(model)
    
    model = get_peft_model(model, peft_config)
    print_trainable_parameters(model)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    train_dataset = FineTuningDataset(os.path.join(dataset_path, "train.jsonl"))
    val_dataset = FineTuningDataset(os.path.join(dataset_path, "val.jsonl"))

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator = lambda data: collate_fn(data, tokenizer=tokenizer, max_length=max_len),
        args=TrainingArguments(
            output_dir="./output/fine_tuned_model",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            evaluation_strategy="steps",
            num_train_epochs=epochs,
            logging_dir="./output/fine_tuned_model/logs",
            save_steps=save_every // batch_size,
            eval_steps=eval_every // batch_size,
            save_total_limit=save_last,
            report_to="wandb",
            run_name=run_name,
            logging_steps=10,
            learning_rate=lr,
        )
    )
    trainer.train()
    output_dir = "./output/model"
    
    torch.save(model, os.path.join(output_dir, final_model_name))
    
    trainer.save_model(output_dir)
    wandb.finish()


if __name__ == "__main__":
    tyro.cli(main)
