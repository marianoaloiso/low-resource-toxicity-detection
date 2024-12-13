from src.data.data_loader import DataLoader
from src.utils.base_experiment import BaseExperiment
from src.utils.model import ModelExperimentMixin
from typing import Dict
import adapters, logging

logger = logging.getLogger(__name__)


class AdapterExperiment(BaseExperiment, ModelExperimentMixin):
    """
    Experiment class for fine-tuning with adapter layers
    Freezes base model parameters and trains only adapter module parameters
    """
    def __init__(self, config_path, experiment_type="adapter_finetuning"):
        super().__init__(experiment_type)
        ModelExperimentMixin.__init__(self, config_path)
        self.adapter_name = "toxicity_adapter"
        self.adapter_identifier = self.config.adapter_identifier

    def setup_adapter_model(
            self,
            model,
    ):
        """
        Setup adapter configuration and prepare model for adapter training

        Args:
            model (transformers.PreTrainedModel): Model instance

        Returns:
            transformers.PreTrainedModel: Model instance with adapter layer
        """

        adapters.init(model)

        model.add_adapter(
            self.adapter_name, 
            config=self.adapter_identifier
        )

        model.train_adapter(self.adapter_name)

        return model

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

            # Apply adapters to the model
            model = self.setup_adapter_model(model)

            # Setup data for the specific context
            datasets = loader.load_language_data(language)

            # Train the model with class weights
            trainer = self.train(
                model,
                language,
                train_dataset=datasets["train"],
                eval_dataset=datasets["validation"],
                use_class_weights=use_class_weights)

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

            # Cleanup GPU memory
            self.cleanup_gpu()
            
            logger.info(f"Completed experiment for language: {language}")

        # Save configuration for the experiment
        self.save_config()

        
if __name__ == "__main__":
    config_path = "configs/adapter_config.yaml"
    languages = ["bodo"]
    experiment = AdapterExperiment(config_path)
    experiment.run_experiment(
        languages=languages, 
        use_class_weights=True
    )

