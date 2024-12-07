from src.data.data_loader import DataLoader
from src.utils.base_experiment import BaseExperiment
from src.utils.model import ModelExperimentMixin
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class BacktranslationExperiment(BaseExperiment, ModelExperimentMixin):
    """Augment training data with backtranslation"""

    def __init__(self, config_path, path_backtranslation_data, experiment_type="backtranslation"):
        super().__init__(config_path, experiment_type=experiment_type)
        ModelExperimentMixin.__init__(self, self.config)
        self.path_backtranslation_data = path_backtranslation_data
    
    def add_backtranslation(self, dataset) -> str:
        """
        Add backtranslation to a given text

        Args:
            dataset (str): Input dataset to add backtranslation to.

        Returns:
            str: Backtranslated text.
        """

        backtranslated_data = pd.read_csv(
            self.path_backtranslation_data,
            usecols=['backtranslated_text', 'label']
        )

        dataset.texts = dataset.texts + backtranslated_data["backtranslated_text"].tolist()
        dataset.labels = dataset.labels + backtranslated_data["label"].tolist()
        
        return dataset

    def run_experiment(self, languages):
        """
        Run the data augmentation experiment
        Overrides base method to include training data augmentation
        """
        data_loader = DataLoader(
            tokenizer=None,
            max_length=self.config.max_length,
        )

        for language in languages:
            logger.info(f"Starting data augmentation experiment for language: {language}")

            # Initialize fresh model for each language
            self.model, self.tokenizer = self.load_automodel(self.model_name, self.num_labels)
            data_loader.tokenizer = self.tokenizer

            # Load language data
            self.datasets = data_loader.load_language_data(language)

            # Augment only training data
            self.datasets['train'] = self.add_backtranslation(self.datasets['train'])

            # Train the model on augmented data
            self.train(language, use_class_weights=True)

            # Evaluate on all datasets
            train_metrics = self.evaluate(language, dataset_split="train")
            val_metrics = self.evaluate(language, dataset_split="validation")
            test_metrics = self.evaluate(language, dataset_split="test")

            # Store results
            results = {
                "train": train_metrics,
                "validation": val_metrics,
                "test": test_metrics
            }
            
            # Save metrics
            self.save_metrics(
                results, 
                save_path=self.metrics_dir / f"{language}_metrics.json"
            )
            
            logger.info(f"Completed backtranslation experiment for language: {language}")

if __name__ == "__main__":
    config_path = "configs/monolingual_finetuning_config.yaml"
    path_backtranslation_data = "data/processed/bodo/backtranslated_train.csv"
    languages = ["bodo"]
    experiment = BacktranslationExperiment(config_path, path_backtranslation_data)
    experiment.run_experiment(languages=languages)

