from src.data.data_loader import DataLoader
from src.utils.base_experiment import BaseExperiment
from src.utils.model import ModelExperimentMixin
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training
)
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class LoRAExperiment(BaseExperiment, ModelExperimentMixin):
    """
    Mixin class to add LoRA (Low-Rank Adaptation) specific model training methods
    """
    def __init__(self, config_path, experiment_type="lora_finetuning"):
        super().__init__(config_path, experiment_type=experiment_type)
        ModelExperimentMixin.__init__(self, self.config)

    def setup_lora_model(
        self, 
        model
    ):
        """
        Setup LoRA configuration and prepare model for low-rank adaptation
        
        Args:
            model: Base model to apply LoRA
        
        Returns:
            PEFT model with LoRA configuration
        """     
        lora_config = LoraConfig(
            r=self.config.r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            bias=self.config.bias,
            task_type=self.config.task_type
        )
        model = prepare_model_for_kbit_training(model)
        lora_model = get_peft_model(model, lora_config)
        return lora_model

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
            model = self.setup_lora_model(model)

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
            
            logger.info(f"Completed LoRA experiment for language: {language}")


if __name__ == "__main__":
    config_path = "configs/lora_config.yaml"
    languages = ["bodo"]
    experiment = LoRAExperiment(config_path)
    experiment.run_experiment(
        languages=languages, 
        use_class_weights=True
    )

