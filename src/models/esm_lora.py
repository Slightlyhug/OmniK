import torch
import torch.nn as nn
from typing import Optional, List
import logging

try:
    from transformers import AutoTokenizer, EsmModel
    from peft import get_peft_model, LoraConfig, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

try:
    import esm
    ESM_FAIR_AVAILABLE = True
except ImportError:
    ESM_FAIR_AVAILABLE = False

logger = logging.getLogger(__name__)

class ESM2LoRAWrapper(nn.Module):
    """
    Problem 3: LoRA E2E Finetuning 
    Loads pretrained ESM-2 model and injects LoRA layers for memory-efficient
    end-to-end multi-task objective optimization. 
    Replaces static H5 sequence feature loading.
    
    Usage:
        seq_encoder = ESM2LoRAWrapper("facebook/esm2_t33_650M_UR50D")
        # OR
        seq_encoder = ESM2LoRAWrapper("/path/to/esm2_t33_650M_UR50D.pt")
        
        embeddings = seq_encoder(batch_of_strings)  # B x L x 1280
    """
    def __init__(
        self, 
        model_name_or_path: str = "facebook/esm2_t33_650M_UR50D", 
        r: int = 8, 
        lora_alpha: int = 32,
        lora_dropout: float = 0.1
    ):
        super().__init__()
        if not PEFT_AVAILABLE:
            raise ImportError(
                "Please install `transformers` and `peft` to use LoRA end-to-end finetuning.\n"
                "  pip install transformers peft"
            )
            
        self.is_fair_esm = model_name_or_path.endswith(".pt")
        
        logger.info(f"Loading ESM backbone for LoRA: {model_name_or_path}")
        
        if self.is_fair_esm:
            if not ESM_FAIR_AVAILABLE:
                raise ImportError("Please install `fair-esm` to load .pt weights (pip install fair-esm)")
            base_model, self.alphabet = esm.pretrained.load_model_and_alphabet_local(model_name_or_path)
            self.batch_converter = self.alphabet.get_batch_converter()
            self.output_dim = base_model.embed_dim
            self.num_layers = base_model.num_layers
            target_modules = ["q_proj", "v_proj"]
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            base_model = EsmModel.from_pretrained(model_name_or_path)
            self.output_dim = int(getattr(base_model.config, "hidden_size", 0) or 0)
            target_modules = ["query", "value"]
        
        # Configure LoRA parameters to update only the attention (query/value) mechanism
        # of the deep layers to prevent catastrophic forgetting of language representations
        peft_config = LoraConfig(
            inference_mode=False,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules  # Critical attention components for motif identification
        )
        
        self.model = get_peft_model(base_model, peft_config)
        self.model.print_trainable_parameters()
        
    def forward(self, sequences: List[str], max_len: int = 150, return_mask: bool = False):
        """
        Dynamically tokenize and compute LoRA features.
        Output: [B, max_len, D]
        """
        if self.is_fair_esm:
            # Truncate sequences to max_len (accounting for <cls> and <eos> tokens, max_len typically refers to sequence length)
            # We'll just truncate the input strings
            truncated_seqs = [s[:max_len-2] if len(s) > max_len-2 else s for s in sequences]
            data = [(str(i), seq) for i, seq in enumerate(truncated_seqs)]
            _, _, batch_tokens = self.batch_converter(data)
            
            # Determine device
            device = next(self.model.parameters()).device
            batch_tokens = batch_tokens.to(device)
            
            # FAIR esm model outputs a dict
            results = self.model(batch_tokens, repr_layers=[self.num_layers], return_contacts=False)
            last_hidden_state = results["representations"][self.num_layers]
            
            if return_mask:
                attention_mask = (batch_tokens != self.alphabet.padding_idx).long()
                return last_hidden_state, attention_mask
            return last_hidden_state
            
        else:
            device = next(self.model.parameters()).device
            inputs = self.tokenizer(
                sequences, 
                padding=True, 
                truncation=True, 
                max_length=max_len, 
                return_tensors="pt"
            ).to(device)
            
            outputs = self.model(**inputs)
            if return_mask:
                return outputs.last_hidden_state, inputs.get("attention_mask")
            return outputs.last_hidden_state

