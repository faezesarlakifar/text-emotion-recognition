import torch
import pandas as pd
from transformers import AutoTokenizer
from dataclasses import dataclass
from itertools import cycle
import config

"""
I will describe my suggestion in three steps:

Combining two (or more) datasets into a single PyTorch Dataset. This dataset will be the input for a PyTorch DataLoader.
Modifying the batch preparation process to produce either one task in each batch or alternatively mix samples from both tasks in each batch.
Handling the highly unbalanced datasets at the batch level by using a batch sampler as part of the DataLoader.

"""



class ArmanEmo(torch.utils.data.Dataset):
    def __init__(self,texts,labels):
        self.texts = texts
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
        self.max_length = config.MAX_LENGTH
        self.labels_dict = config.LABEL_DICT_MAPPING

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )
        inputs = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label,dtype=torch.long)
        }
        return inputs


class EmoPars(torch.utils.data.Dataset):
    def __init__(self,texts, labels):
        self.texts = texts
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
        self.max_length = config.MAX_LENGTH

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )

        inputs = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.float32)  # Use the provided numeric label directly
        }

        return inputs
    




def multitask_dataloader(dataset1,dataset2,batch_size,train=True):
    
    dataloader1 = torch.utils.data.DataLoader(
        dataset=dataset1,
        batch_size=batch_size//2,
        shuffle=train,
    )
    dataloader2 = torch.utils.data.DataLoader(
        dataset=dataset2,
        batch_size=batch_size//2,
        shuffle=train,
    )

    dataloader = zip(dataloader1,cycle(dataloader2))

    return dataloader