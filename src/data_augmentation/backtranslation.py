from src.data.data_loader import DataLoader
from src.utils.base_experiment import BaseExperiment
from src.utils.model import ModelExperimentMixin
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class BacktranslationExperiment(BaseExperiment, ModelExperimentMixin):
    """Augment training data with backtranslation"""

    def __init__(self, config_path, path_backtranslation_data, experiment_type="backtranslation"):
        super().__init__(experiment_type)
        ModelExperimentMixin.__init__(self, config_path)
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
        data_loader = DataLoader(
            tokenizer=None,
            max_length=self.config.max_length,
        )

        for language in languages:
            logger.info(f"Starting data augmentation experiment for language: {language}")

            # Initialize fresh model for each language
            model, tokenizer = self.load_automodel(self.model_name, self.num_labels)
            data_loader.tokenizer = tokenizer

            # Load language data
            datasets = data_loader.load_language_data(language)

            # Augment only training data
            datasets['train'] = self.add_backtranslation(datasets['train'])

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
            
            logger.info(f"Completed backtranslation experiment for language: {language}")

if __name__ == "__main__":
    config_path = "configs/monolingual_finetuning_config.yaml"
    path_backtranslation_data = "data/processed/bodo/backtranslated_train.csv"
    languages = ["bodo"]
    experiment = BacktranslationExperiment(config_path, path_backtranslation_data)
    experiment.run_experiment(languages=languages)

