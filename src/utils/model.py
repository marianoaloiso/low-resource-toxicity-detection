from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Optional, Dict, Any

def load_automodel(model_name, num_labels):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return model, tokenizer

@dataclass
class ModelConfig:
    """Configuration for model training and inference"""
    # Model architecture
    model_name: str = None
    max_length: int = 512
    hidden_size: Optional[int] = None
    num_labels: int = 2
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_steps: int = 0
    
    # Other settings
    device: str = "cpu"
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ModelConfig":
        return cls(**{
            k: v for k, v in config_dict.items() 
            if k in cls.__dataclass_fields__
        })
    
