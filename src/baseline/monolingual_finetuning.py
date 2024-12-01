from src.data.data_loader import DataLoader
from src.project_setup import ProjectSetup
from src.utils.base_experiment import BaseExperiment
from src.utils.model import calculate_class_weights, load_automodel, WeightedTrainer
from transformers import Trainer, TrainingArguments
import logging

logger = logging.getLogger(__name__)

class MonolingualFinetuningExperiment(BaseExperiment):
    """Fine-tuning model on individual datasets experiment"""

    def __init__(self, config_path, experiment_type="monolingual_finetuning"):
        super().__init__(config_path, experiment_type=experiment_type)
        self.model_name = self.config.model_name
        self.num_labels = self.config.num_labels
        self.model = None
        self.tokenizer = None

    def train(self, language, use_class_weights=False):
        """Train the model on the training data"""
        logger.info(f"Starting training for language: {language}")

        class_weights = None
        if use_class_weights:
            class_weights = calculate_class_weights(self.datasets["train"].labels)

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

        self.save_json(
            train_results.metrics,
            model_save_path / f"training_metrics.json"
        )

        logger.info(f"Completed training for language: {language}")

    def evaluate(self, language, dataset_split="test"):
        """Evaluate the model on the specified dataset split"""
        logger.info(f"Evaluating model on {dataset_split} set")
        
        trainer = Trainer(
            model=self.model,
            args=TrainingArguments(output_dir=ProjectSetup.DUMMY_DIR)
        )
        
        predictions = trainer.predict(self.datasets[dataset_split])
        
        pred_labels = predictions.predictions.argmax(-1)
        true_labels = [x["labels"].item() for x in self.datasets[dataset_split]]
        
        self.save_predictions(
            predictions=pred_labels.tolist(),
            true_labels=true_labels,
            save_path=self.predictions_dir / f"{language}_predictions_{dataset_split}.json"
        )
        
        metrics = self.calculate_metrics(
            true_labels=true_labels,
            predictions=pred_labels.tolist()
        )
        
        return metrics
    
    def run_experiment(self, languages):
        """Run the fine-tuning experiment"""
        data_loader = DataLoader(
            tokenizer=None,
            max_length=self.config.max_length,
        )

        for language in languages:
            logger.info(f"Starting experiment for language: {language}")

            # Initialize fresh model for each language
            self.model, self.tokenizer = load_automodel(self.model_name, self.num_labels)
            data_loader.tokenizer = self.tokenizer

            # Setup data for the language
            self.datasets = data_loader.load_language_data(language)

            # Train the model
            self.train(language, use_class_weights=True)

            # Evaluate on training set
            train_metrics = self.evaluate(language, dataset_split="train")

            # Evaluate on validation set
            val_metrics = self.evaluate(language, dataset_split="validation")

            # Evaluate on test set
            test_metrics = self.evaluate(language, dataset_split="test")

            # Combine all metrics
            all_metrics = {
                "train": train_metrics,
                "validation": val_metrics,
                "test": test_metrics
            }
            
            # Save metrics for this language
            self.save_metrics(
                all_metrics, 
                save_path=self.metrics_dir / f"{language}_metrics.json"
            )
            
            logger.info(f"Completed experiment for language: {language}")


if __name__ == "__main__":
    config_path = "configs/monolingual_finetuning_config.yaml"
    languages = ["bodo"]
    experiment = MonolingualFinetuningExperiment(config_path)
    experiment.run_experiment(languages=languages)

