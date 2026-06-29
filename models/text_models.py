import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class TextFeatureExtractor(nn.Module):
    """
    Wrapper for Pre-trained Text Models (PhoBERT, XLM-R, ViSoBERT)
    """
    def __init__(self, model_name: str, hidden_dim: int = 768):
        super().__init__()
        self.model_name = model_name
        # Support for specific github paths or HuggingFace hub names
        # if ViSoBERT is specified by huggingface path (e.g. `uitnlp/visobert` if user uploads it, 
        # or we just load it as a standard model if it's compatible)
        # Note: If it's a local path from clone, model_name should point to the directory.
        
        try:
            self.encoder = AutoModel.from_pretrained(model_name)
            if hasattr(self.encoder.config, 'use_cache'):
                self.encoder.config.use_cache = False
        except Exception as e:
            print(f"Failed to load {model_name} from HuggingFace. Trying as standard AutoModel... Error: {e}")
            # Fallback to phobert if strictly needed
            if 'visobert' in model_name.lower():
                print("Falling back to vinai/phobert-base-v2 for text encoder due to missing ViSoBERT weights path.")
                self.encoder = AutoModel.from_pretrained("vinai/phobert-base-v2")
                if hasattr(self.encoder.config, 'use_cache'):
                    self.encoder.config.use_cache = False
            else:
                raise e

        # Extract hidden size
        config = AutoConfig.from_pretrained(model_name) if 'visobert' not in model_name.lower() else AutoConfig.from_pretrained("vinai/phobert-base-v2")
        self.encoder_dim = config.hidden_size
        
        self.proj = nn.Linear(self.encoder_dim, hidden_dim)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use pooled output or mean pooling over last hidden state
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            # Mean pooling
            last_hidden = outputs.last_hidden_state
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            pooled = torch.sum(last_hidden * mask_expanded, 1) / torch.clamp(mask_expanded.sum(1), min=1e-9)
            
        features = self.proj(pooled)
        return features
