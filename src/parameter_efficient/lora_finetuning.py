from src.utils.base_experiment import BaseExperiment
from src.utils.model import ModelExperimentMixin, WeightedTrainer, ModelConfig
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training
)
from transformers import TrainingArguments
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class MonolingualLoRAExperiment(BaseExperiment, ModelExperimentMixin):
    """
    Mixin class to add LoRA (Low-Rank Adaptation) specific model training methods
    """
    def __init__(self, config_path, experiment_type="monolingual_lora_finetuning"):
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

    def train(
        self, 
        language, 
        use_class_weights=False
    ):
        """
        Modified training method to incorporate LoRA
        
        Args:
            language (str): Language to train on
            use_class_weights (bool): Whether to apply class weights
        """
        logger.info(f"Starting LoRA training for language: {language}")

        # Prepare class weights if needed
        class_weights = None
        if use_class_weights:
            class_weights = self.calculate_class_weights(self.datasets["train"].labels)

        # Apply LoRA to the model
        self.model = self.setup_lora_model(self.model)

        # Compute model trainable parameters percentage
        trainable_params, total_params = self._count_parameters(self.model)
        trainable_percentage = (trainable_params / total_params) * 100
        logger.info(f"Trainable Parameters: {trainable_params} ({trainable_percentage:.2f}%)")
        logger.info(f"Total Parameters: {total_params}")

        # Rest of the training remains similar to the original implementation
        model_save_path = self.models_dir / f"{language}_lora_model"

        training_args = TrainingArguments(
            output_dir=model_save_path,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            logging_dir=self.logs_dir,
            logging_steps=10,
            eval_strategy="epoch"
        )

        trainer = WeightedTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.datasets["train"],
            eval_dataset=self.datasets["validation"],
            class_weights=class_weights
        )

        # Start training with performance tracking
        train_results = trainer.train()

        # Save the LoRA model
        trainer.save_model(str(model_save_path))

        # Save training metrics including computational efficiency
        metrics = train_results.metrics
        metrics.update({
            "total_params": total_params,
            "trainable_params": trainable_params,
            "trainable_percentage": trainable_percentage
        })

        self.save_metrics(
            metrics,
            model_save_path / f"training_metrics.json"
        )

        logger.info(f"Completed LoRA training for language: {language}")

    def _count_parameters(self, model):
        """
        Count total and trainable parameters in the model
        
        Args:
            model: PyTorch model
        
        Returns:
            Tuple of (trainable_params, total_params)
        """
        trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        total_params = sum(p.numel() for p in model.parameters())
        return trainable_params, total_params


if __name__ == "__main__":
    config_path = "configs/monolingual_lora_config.yaml"
    languages = ["bodo"]
    experiment = MonolingualLoRAExperiment(config_path)
    experiment.run_experiment(
        languages=languages, 
        use_class_weights=True
    )

