from src.data.data_loader import DataLoader
from src.project_setup import ProjectSetup
from src.utils.base_experiment import BaseExperiment
from src.utils.model import calculate_class_weights, load_automodel, WeightedTrainer
from transformers import TrainingArguments
import logging

logger = logging.getLogger(__name__)

class CrosslingualTransferExperiment(BaseExperiment):
    """Crosslingual transfer learning experiment for low-resource languages"""

    def __init__(self, config_path):
        super().__init__(config_path, experiment_type="crosslingual_transfer")
        self.model_name = self.config.model_name
        self.num_labels = self.config.num_labels
        self.model = None
        self.tokenizer = None
        self.available_languages = ProjectSetup.LANGUAGES

    def train_on_languages(self, source_languages, use_class_weights=False):
        """
        Train the model on specified source languages
        """
        logger.info(f"Training on source languages: {source_languages}")
        
        combined_datasets = self.data_loader.load_combined_language_data(
            languages=source_languages,
            splits=["train", "validation"]
        )

        class_weights = None
        if use_class_weights:
            class_weights = calculate_class_weights(combined_datasets["train"].labels)

        model_save_path = self.models_dir / f"crosslingual_model_{'_'.join(source_languages)}"
        
        training_args = TrainingArguments(
            output_dir=model_save_path,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
        )

        trainer = WeightedTrainer(
            model=self.model,
            args=training_args,
            train_dataset=combined_datasets["train"],
            eval_dataset=combined_datasets["validation"],
            class_weights=class_weights,
        )

        train_results = trainer.train()
        trainer.save_model(str(model_save_path))

        self.save_json(
            train_results.metrics,
            model_save_path / "training_metrics.json",
        )

        return trainer

    def evaluate_on_language(self, trainer, target_language, dataset_split="test"):
        """
        Evaluate the model on a specific target language
        
        Args:
            trainer: Trained model trainer
            target_language (str): Language to evaluate on
            dataset_split (str, optional): Dataset split to evaluate. Defaults to "test".
        
        Returns:
            dict: Evaluation metrics
        """
        logger.info(f"Evaluating on target language: {target_language}")
        
        target_datasets = self.data_loader.load_language_data(target_language)

        predictions = trainer.predict(target_datasets[dataset_split])
        
        pred_labels = predictions.predictions.argmax(-1)
        true_labels = [x["labels"].item() for x in target_datasets[dataset_split]]
        
        self.save_predictions(
            predictions=pred_labels.tolist(),
            true_labels=true_labels,
            save_path=self.predictions_dir / f"{target_language}_predictions_{dataset_split}.json"
        )
        
        metrics = self.calculate_metrics(
            true_labels=true_labels,
            predictions=pred_labels.tolist()
        )
        
        return metrics

    def run_experiment(self, languages):
        """
        Run crosslingual transfer learning experiment
        Iterate through all possible source language combinations
        Train on two languages and evaluate on the third
        """
        self.model, self.tokenizer = load_automodel(self.model_name, self.num_labels)

        self.data_loader = DataLoader(
            tokenizer=self.tokenizer,
            max_length=self.config.max_length,
        )

        for target_language in languages:
            source_languages = [lang for lang in self.available_languages if lang != target_language]
            
            logger.info(f"Experiment: Train on {source_languages}, Evaluate on {target_language}")

            trainer = self.train_on_languages(source_languages, use_class_weights=True)

            metrics = {
                "train": self.evaluate_on_language(trainer, target_language, "train"),
                "validation": self.evaluate_on_language(trainer, target_language, "validation"),
                "test": self.evaluate_on_language(trainer, target_language, "test")
            }

            self.save_metrics(
                metrics, 
                save_path=self.metrics_dir / f"{target_language}_metrics.json"
            )

            logger.info(f"Completed evaluation for language: {target_language}")

        logger.info("Completed crosslingual transfer experiment")


if __name__ == "__main__":
    config_path = "configs/crosslingual_transfer_config.yaml"
    languages = ["bodo"]
    experiment = CrosslingualTransferExperiment(config_path)
    experiment.run_experiment(languages)

