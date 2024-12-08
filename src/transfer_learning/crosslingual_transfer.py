from src.data.data_loader import DataLoader
from src.utils.model import ModelExperimentMixin
from src.project_setup import ProjectSetup
from src.utils.base_experiment import BaseExperiment
import logging

logger = logging.getLogger(__name__)

class CrosslingualTransferExperiment(BaseExperiment, ModelExperimentMixin):
    """Crosslingual transfer learning experiment for low-resource languages"""

    def __init__(self, config_path):
        super().__init__(config_path, experiment_type="crosslingual_transfer")
        ModelExperimentMixin.__init__(self, self.config)
        self.available_languages = ProjectSetup.LANGUAGES

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

        loader = DataLoader(
            tokenizer=tokenizer,
            max_length=self.config.max_length,
        )

        for target_language in languages:
            source_languages = [lang for lang in self.available_languages if lang != target_language]
            
            logger.info(f"Experiment: Train on {source_languages}, Evaluate on {target_language}")

            combined_datasets = loader.load_combined_language_data(
                languages=source_languages,
                splits=["train", "validation"]
            )
            trainer = self.train(
                model,
                source_languages,
                train_dataset=combined_datasets["train"],
                eval_dataset=combined_datasets["validation"],
                use_class_weights=use_class_weights
            )
    
            datasets = loader.load_language_data(target_language)
            all_metrics = {}
            for split in evaluate_splits:
                logger.info(f"Evaluating model on {split} set")
                metrics = self.evaluate(
                    trainer,
                    datasets[split],
                    target_language,
                    dataset_split=split
                )
                all_metrics[split] = metrics

            self.save_metrics(all_metrics, f"{target_language}_metrics.json")

            logger.info(f"Completed evaluation for language: {target_language}")

        logger.info("Completed crosslingual transfer experiment")


if __name__ == "__main__":
    config_path = "configs/crosslingual_transfer_config.yaml"
    languages = ["bodo"]
    experiment = CrosslingualTransferExperiment(config_path)
    experiment.run_experiment(languages)

