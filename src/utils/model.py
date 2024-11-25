from dataclasses import dataclass
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from typing import Optional, Dict, Any
import torch

def load_automodel(model_name, num_labels):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return model, tokenizer

def calculate_class_weights(labels: list) -> torch.Tensor:
    """Calculate class weights for imbalanced datasets based on the frequency of each class"""
    class_weights = {}
    total = len(labels)
    for label in set(labels):
        class_weights[label] = total / (labels.count(label) * len(set(labels)))
    return torch.tensor(list(class_weights.values()))


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
    

class WeightedTrainer(Trainer):
    """Custom Trainer class for handling class weights during training"""
    def __init__(self, *args, class_weights: Optional[torch.Tensor] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        **kwargs
    ):
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs.logits
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(model.device)
        loss_fct = CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss
    
