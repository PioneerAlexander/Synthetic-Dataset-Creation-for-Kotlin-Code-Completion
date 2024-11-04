"""
    Tokenized dataset for passing to the PEFT Trainer
"""
from typing import List, Dict

from torch.utils.data import Dataset

import json

class FineTuningDataset(Dataset):
    """
        Every element is fully tokenized task code
    """

    def __init__(self, path_to_dataset: str):
        self.dataset: list[dict] = []
        with open(path_to_dataset, "r") as f:
            content = f.read()
            for line in content.split("\n"):
                if line:
                    self.dataset.append(json.loads(line))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> str:
        return self.dataset[idx]["code"] + "<SEP>" + self.dataset[idx]["prompt"]
