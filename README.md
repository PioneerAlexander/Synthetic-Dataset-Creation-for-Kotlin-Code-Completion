# Synthetic Dataset Creation for Kotlin Code Completion
**JetBrains test task**

## Experimental setup

Install the requirements for the project as:
```bash
chmod +x setup.sh
./setup.sh
```
You can comment the lines, if you have already some dependencies installed. 


## Save locally the model you are planning to fine-tune

For example, for codegen-350M-mono model:
```bash
python save_model.py --model_name="Salesforce/codegen-350M-mono" --model_save_name="codegen-350M-mono.pth" --tokenizer_save_path_name="codegen-350M-mono-tokenizer/"
```

For granite-3b-code-base model:
```bash
python save_model.py --model_name="ibm-granite/granite-3b-code-base-2k" --model_save_name="granite.pth" --tokenizer_save_path_name="granite-tokenizer/"
```

## The dataset creation and preprocessing:

For obtain a synthetic dataset, run the following script. The dataset was generated using ChatGPT API, so for reproducing the experiment, you will need to configure a OpenAI token.

```bash
python generate_dataset.py --dataset_save_name="./data/kotlin/synthetic_dataset_7k.json" --dataset_size=7000
```
Preprocess the dataset for the `train.jsonl`, `val.jsonl` form, needed for finetuning.
```bash
python preprocess_dataset.py --generated-synthetic-dataset-filename=data/kotlin/synthetic_dataset_7k.json --dataset-save-path=./data/kotlin
```

## Model finetuning pipeline

For finetuning the model, run the following (example for codegen model):

```bash
python -m finetune codegen-350M-mono.pth codegen-350M-mono-tokenizer/ data/kotlin/ --batch_size=1
```

## Evaluating models

For evaluating the models, run the following (example for granite model):

```bash
python eval_model.py granite.pth granite-tokenizer/ granite-pretrained
```

## Analysis part
Analysis part is located in the file [report.md](report.md)


