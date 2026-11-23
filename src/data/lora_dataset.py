import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional

class LoRAPTMDataset(Dataset):
    """
    Replaces H5 DataLoader with raw FASTA string loading for end-to-end ESM-2 LoRA fine-tuning.
    """
    def __init__(self, records: List[Dict[str, Any]], ptm_types: List[str], window_size: int, tokenizer):
        self.records = records
        self.ptm_types = ptm_types
        self.window_size = window_size
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        seq = record.get('sequence', '')
        # Basic tokenization
        tokens = self.tokenizer(seq, return_tensors='pt', padding='max_length', max_length=self.window_size, truncation=True)
        
        # Build multi-label target
        targets = torch.zeros(len(self.ptm_types))
        for i, ptm in enumerate(self.ptm_types):
            if ptm in record.get('labels', []):
                targets[i] = 1.0
                
        return {
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
            'targets': targets,
            'id': record.get('id', str(idx))
        }
