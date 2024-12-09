from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity
from src.data.data_loader import DataLoader
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification, 
    Trainer, TrainingArguments
)
from typing import Optional, Dict, Any
import logging, torch, yaml

logger = logging.getLogger(__name__)


class ModelExperimentMixin:
    """
    Mixin class to add generic model training and evaluation methods
    to existing BaseExperiment classes
    """
    def __init__(self, config_path=None):
        with open(config_path) as f:
            self.config = ModelConfig.from_dict(yaml.safe_load(f))
        self.model_name = getattr(self.config, 'model_name', None)
        self.num_labels = getattr(self.config, 'num_labels', None)
            
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

    def count_parameters(self, model: torch.nn.Module):
        """
        Count total and trainable parameters in the model
        
        Args:
            model (torch.nn.Module): Model instance
        
        Returns:
            Tuple of (trainable_params, total_params)
        """
        trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        total_params = sum(p.numel() for p in model.parameters())
        return trainable_params, total_params

    def train(
        self, 
        model,
        language,
        train_dataset,
        eval_dataset,
        use_class_weights=False
    ):
        """
        Train the model on the training data
        
        Args:
            model (torch.nn.Module): Model instance
            language (str): Language to train on
            use_class_weights (bool): Whether to apply class weights
        
        Returns:
            Training results
        """
        logger.info(f"Starting training for language: {language}")

        trainable_params, total_params = self.count_parameters(model)
        trainable_percentage = (trainable_params / total_params) * 100
        logger.info(f"Trainable Parameters: {trainable_params} ({trainable_percentage:.2f}%)")
        logger.info(f"Total Parameters: {total_params}")

        class_weights = None
        if use_class_weights:
            class_weights = self.calculate_class_weights(train_dataset.labels)

        training_args = TrainingArguments(
            output_dir=self.models_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
        )

        trainer = WeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            class_weights=class_weights
        )

        train_results = trainer.train()
        
        trainer.save_model()

        metrics = train_results.metrics
        metrics.update({
            "total_params": total_params,
            "trainable_params": trainable_params,
            "trainable_percentage": trainable_percentage
        })
        self.save_training_metrics(metrics)

        logger.info(f"Completed training for language: {language}")

        return trainer

    def evaluate(self, trainer, dataset, language, dataset_split=""):
        """
        Evaluate the model on the specified dataset split
        
        Args:
            trainer (Trainer): Trained model trainer
            dataset (Dataset): Dataset to evaluate on
            language (str): Language to evaluate on
        
        Returns:
            Evaluation metrics
        """

        predictions = trainer.predict(dataset)

        logits = predictions.predictions
        pred_labels = logits.argmax(-1)
        true_labels = [x["labels"].item() for x in dataset]
        
        prediction_filename = f"{language}_{dataset_split}_predictions.json"
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
            model, tokenizer = self.load_automodel(
                self.model_name, 
                self.num_labels
            )
            loader.tokenizer = tokenizer

            # Setup data for the specific context
            datasets = loader.load_language_data(language)

            # Train the model with class weights
            trainer = self.train(
                model,
                language,
                train_dataset=datasets["train"],
                eval_dataset=datasets["validation"],
                use_class_weights=use_class_weights
            )

            # Evaluation across different splits
            all_metrics = {}
            for split in evaluate_splits:
                logger.info(f"Evaluating model on {split} set")
                metrics = self.evaluate(
                    trainer,
                    datasets[split],
                    language,
                    dataset_split=split
                )
                all_metrics[split] = metrics

            # Save metrics for this iteration
            self.save_metrics(all_metrics, f"{language}_metrics.json")
            
            logger.info(f"Completed experiment for language: {language}")


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

