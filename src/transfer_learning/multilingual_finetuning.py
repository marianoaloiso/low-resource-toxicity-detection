from src.data.data_loader import DataLoader
from src.utils.base_experiment import BaseExperiment
from src.utils.model import ModelExperimentMixin
import logging

logger = logging.getLogger(__name__)

class MultilingualFinetuningExperiment(BaseExperiment, ModelExperimentMixin):
    """Fine-tuning model on multilingual dataset"""

    def __init__(self, config_path):
        super().__init__(experiment_type="multilingual_finetuning")
        ModelExperimentMixin.__init__(self, config_path)

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
        model, tokenizer = self.load_automodel(self.model_name, self.num_labels)

        data_loader = DataLoader(
            tokenizer=tokenizer,
            max_length=self.config.max_length,
        )

        datasets = data_loader.load_combined_language_data()

        trainer = self.train(
            model,
            "combined",
            train_dataset=datasets["train"],
            eval_dataset=datasets["validation"],
            use_class_weights=use_class_weights
        )

        for language in languages:
            logger.info(f"Evaluating experiment for language: {language}")

            self.datasets = data_loader.load_language_data(language)

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

        logger.info("Completed multilingual experiment")


if __name__ == "__main__":
    config_path = "configs/multilingual_finetuning_config.yaml"
    languages = ["bodo"]
    experiment = MultilingualFinetuningExperiment(config_path)
    experiment.run_experiment(languages)

