from src.data.data_loader import DataLoader
from src.utils.base_experiment import BaseExperiment
from src.utils.model import ModelExperimentMixin
import logging

logger = logging.getLogger(__name__)

class MultilingualFinetuningExperiment(BaseExperiment, ModelExperimentMixin):
    """Fine-tuning model on multilingual dataset"""

    def __init__(self, config_path):
        super().__init__(config_path, experiment_type="multilingual_finetuning")
        ModelExperimentMixin.__init__(self, self.config)

    def run_experiment(self, languages):
        """Run the multilingual fine-tuning experiment"""
        self.model, self.tokenizer = self.load_automodel(self.model_name, self.num_labels)

        data_loader = DataLoader(
            tokenizer=self.tokenizer,
            max_length=self.config.max_length,
        )

        self.datasets = data_loader.load_combined_language_data()

        trainer = self.train("combined")

        for language in languages:
            logger.info(f"Evaluating experiment for language: {language}")

            self.datasets = data_loader.load_language_data(language)

            train_metrics = self.evaluate(trainer, language, dataset_split="train")

            val_metrics = self.evaluate(trainer, language, dataset_split="validation")

            test_metrics = self.evaluate(trainer, language, dataset_split="test")

            all_metrics = {
                "train": train_metrics,
                "validation": val_metrics,
                "test": test_metrics
            }
            
            self.save_metrics(
                all_metrics, 
                save_path=self.metrics_dir / f"{language}_metrics.json"
            )
            
            logger.info(f"Completed evaluation for language: {language}")

        logger.info("Completed multilingual experiment")


if __name__ == "__main__":
    config_path = "configs/multilingual_finetuning_config.yaml"
    languages = ["bodo"]
    experiment = MultilingualFinetuningExperiment(config_path)
    experiment.run_experiment(languages)

