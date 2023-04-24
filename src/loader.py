import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from transformers import Wav2Vec2FeatureExtractor

class Custom_Dataset(Dataset):
    def __init__(self, df, tokenizer=None, feature_extractor=None):
        self.text = df['text']
        self.wav = df['speech_array'].values.tolist()
        self.target_sampling_rate = 16000
        self.labels = df['Emotion']
        self.tokenizer = tokenizer
        
        self.feature_extractor = feature_extractor(self.wav, sampling_rate=self.target_sampling_rate)

    def __len__(self): 
        return len(self.labels)

    
    def __getitem__(self, idx):
        text = self.text[idx]
        label = self.labels[idx]
        input_values = self.feature_extractor['input_values'][idx]
        
        tokenized = self.tokenizer(text=text,
                                   padding='max_length',
                                   truncation=True,
                                   max_length=256,
                                   return_attention_mask=True,
                                   return_token_type_ids=True,
                                   return_tensors='pt')
        
        data = {
                'input_values' : input_values,
                'tokenized_input_ids' : tokenized['input_ids'].squeeze(),
                'tokenized_attention_mask' : tokenized['attention_mask'].squeeze(),
                'tokenized_token_type_ids' : tokenized['token_type_ids'].squeeze(),
                'labels' : torch.tensor(label, dtype=torch.long),
                }

        return data

@dataclass
class DataCollatorCTCWithPadding:
    feature_extractor: Wav2Vec2FeatureExtractor
    padding: Union[bool, str] = True
    truncation: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]

        d_type_ls = [features[0]['tokenized_input_ids'].dtype, 
                     features[0]['tokenized_attention_mask'].dtype,
                     features[0]['tokenized_token_type_ids'].dtype,
                     features[0]['labels'].dtype]

        batch = self.feature_extractor.pad(
            input_features,
            padding=self.padding,
            truncation=self.truncation,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch['tokenized_input_ids'] = torch.stack([feature["tokenized_input_ids"] for feature in features], dim=0)
        batch['tokenized_attention_mask'] = torch.stack([feature["tokenized_attention_mask"] for feature in features], dim=0)
        batch['tokenized_token_type_ids'] = torch.stack([feature["tokenized_token_type_ids"] for feature in features], dim=0)
        batch['labels'] = torch.tensor([feature["labels"] for feature in features], dtype=d_type_ls[3])

        return batch