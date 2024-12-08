from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity
from src.data.data_loader import DataLoader
from src.project_setup import ProjectSetup
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification, 
    Trainer, TrainingArguments
)
from typing import Optional, Dict, Any
import logging, torch

logger = logging.getLogger(__name__)


class ModelExperimentMixin:
    """
    Mixin class to add generic model training and evaluation methods
    to existing BaseExperiment classes
    """
    def __init__(self, config=None):
        self.config = config
        self.model_name = getattr(config, 'model_name', None)
        self.num_labels = getattr(config, 'num_labels', None)
        self.model = None
        self.tokenizer = None

    def load_automodel(self, model_name, num_labels):
        """
        Load a pre-trained automodel and tokenizer from Hugging Face model hub

        Args:
            model_name (str): Model name or path
            num_labels (int): Number of output labels

        Returns:
            Tuple: Model and tokenizer instances
        """
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        return model, tokenizer

    def calculate_class_weights(self, labels: list) -> torch.Tensor:
        """
        Calculate class weights for imbalanced datasets based on the frequency of each class

        Args:
            labels (list): List of class labels

        Returns:
            torch.Tensor: Class weights
        """
        class_weights = {}
        total = len(labels)
        for label in set(labels):
            class_weights[label] = total / (labels.count(label) * len(set(labels)))
        return torch.tensor(list(class_weights.values()))

    def count_parameters(self) -> tuple:
        """
        Count total and trainable parameters in the model
        
        Args:
            model: PyTorch model
        
        Returns:
            Tuple of (trainable_params, total_params)
        """
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        total_params = sum(p.numel() for p in self.model.parameters())
        return trainable_params, total_params

    def train(
        self, 
        language, 
        use_class_weights=False
    ):
        """
        Train the model on the training data
        
        Args:
            language (str): Language to train on
            use_class_weights (bool): Whether to apply class weights
        
        Returns:
            Training results
        """
        logger.info(f"Starting training for language: {language}")

        class_weights = None
        if use_class_weights:
            class_weights = self.calculate_class_weights(self.datasets["train"].labels)

        model_save_path = self.models_dir / f"{language}_model" 

        training_args = TrainingArguments(
            output_dir=model_save_path,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
        )

        trainer = WeightedTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.datasets["train"],
            eval_dataset=self.datasets["validation"],
            class_weights=class_weights
        )

        train_results = trainer.train()
        
        trainer.save_model(str(model_save_path))

        # Save training metrics
        self.save_metrics(
            train_results.metrics,
            model_save_path / f"training_metrics.json"
        )

        logger.info(f"Completed training for language: {language}")

    def evaluate(
        self, 
        language, 
        dataset_split="test"
    ):
        """
        Evaluate the model on the specified dataset split
        
        Args:
            language (str): Language to evaluate on
            dataset_split (str): Which dataset split to evaluate on
        
        Returns:
            Evaluation metrics
        """
        logger.info(f"Evaluating model on {dataset_split} set")

        trainer = Trainer(
            model=self.model,
            args=TrainingArguments(output_dir=ProjectSetup.DUMMY_DIR)
        )
        
        predictions = trainer.predict(self.datasets[dataset_split])

        logits = predictions.predictions
        pred_labels = logits.argmax(-1) #predictions.predictions.argmax(-1)
        true_labels = [x["labels"].item() for x in self.datasets[dataset_split]]
        
        prediction_filename = f"{language}_predictions_{dataset_split}.json"
        self.save_predictions(
            predictions=pred_labels.tolist(),
            true_labels=true_labels,
            save_path=self.predictions_dir / prediction_filename,
            logits=logits.tolist()
        )

        metrics = self.calculate_metrics(
            true_labels=true_labels,
            predictions=pred_labels.tolist()
        )
        
        return metrics

    def run_experiment(
        self, 
        languages: list,
        use_class_weights: bool = True,
        evaluate_splits: list = ["train", "validation", "test"]
    ):
        """
        Run the full experiment pipeline
        
        Args:
            languages (list): List of languages to process
            use_class_weights (bool): Whether to apply class weights during training
            evaluate_splits (list): Which dataset splits to evaluate and save metrics for
        """
        loader = DataLoader(
            tokenizer=None,
            max_length=self.config.max_length,
        )
        
        for language in languages:
            logger.info(f"Starting experiment for language: {language}")

            # Initialize model for each language
            self.model, self.tokenizer = load_automodel(
                self.model_name, 
                self.num_labels
            )
            loader.tokenizer = self.tokenizer

            # Setup data for the specific context
            self.datasets = loader.load_language_data(language)

            # Train the model with class weights
            self.train(language, use_class_weights=use_class_weights)

            # Evaluation across different splits
            all_metrics = {}
            for split in evaluate_splits:
                metrics = self.evaluate(language, dataset_split=split)
                all_metrics[split] = metrics

            # Save metrics for this iteration
            metrics_filename = f"{language}_metrics.json"
            self.save_metrics(
                all_metrics, 
                save_path=self.metrics_dir / metrics_filename
            )
            
            logger.info(f"Completed experiment for language: {language}")

# TODO: Refactor the following methods to be part of the ModelExperimentMixin class
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
    # Fields common to all experiments
    num_labels: int = 2
    device: str = "cpu"

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ModelConfig":
        # First, extract predefined fields
        known_fields = {k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__}
        
        # Create the instance with known fields
        config_instance = cls(**known_fields)
        
        # Dynamically add additional attributes
        for key, value in config_dict.items():
            if key not in cls.__dataclass_fields__:
                setattr(config_instance, key, value)
        
        return config_instance
    

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
    

class CustomNLIPipeline:
    """Custom pipeline for zero-shot NLI using Indic-BERT"""

    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def encode_text(self, text):
        """Generate embeddings for the input text"""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings

    def classify(self, text, candidate_labels, hypothesis_template="This text is {}."):
        """Perform zero-shot classification"""
        text_embedding = self.encode_text(text)

        hypotheses = [hypothesis_template.format(label) for label in candidate_labels]
        label_embeddings = torch.stack([self.encode_text(h) for h in hypotheses])

        similarities = cosine_similarity(
            text_embedding.detach().numpy(),
            label_embeddings.squeeze(1).detach().numpy()
        )[0]

        results = {label: score for label, score in zip(candidate_labels, similarities)}
        return results

