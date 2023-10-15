import copy
import json
import os
import torch

from sentencepiece import SentencePieceProcessor
from torch.utils.data import Dataset
from typing import List

TRAIN_TEST_SPLIT = 4000


class SafetyEvaluationDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train", max_words=2048):
        self.ann = []
        with open(dataset_config.data_path, "r", encoding="utf-8") as input_fp:
            for line in input_fp:
                question = json.loads(line)
                guidelines = question["prompt"][1]["content"]
                last_question = question["prompt"][-1]["content"]
                evaluation = question["response"]["content"]
                prompt = guidelines + "\n" + "Conversation:\n" + last_question
                sample = {
                    "prompt" : prompt,
                    "evaluation" : evaluation
                }
                self.ann.append(sample)
            
        if partition == "train":
            self.ann = self.ann[:TRAIN_TEST_SPLIT]
        else:
            self.ann = self.ann[TRAIN_TEST_SPLIT:]

        self.max_words = max_words
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss
        sample = self.ann[index]
        
        prompt = sample["prompt"]
        evaluation = sample["evaluation"]
        example = prompt + "\n" + evaluation
        prompt = torch.tensor(
            self.tokenizer.encode(prompt), dtype=torch.int64
        )
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(
            example, dtype=torch.int64
        )
        
        padding = self.max_words - example.shape[0]
        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            example = example[: self.max_words]   
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = IGNORE_INDEX
        example_mask = example_mask.float()
        label_mask = label_mask.float()

        return {
            "input_ids": example,
            "labels": labels,
            "attention_mask":example_mask,
        }
  