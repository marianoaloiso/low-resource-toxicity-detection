from src.utils.base_experiment import BaseExperiment
from src.utils.model import ModelExperimentMixin, WeightedTrainer
from transformers import TrainingArguments
from typing import Dict
import adapters, logging

logger = logging.getLogger(__name__)


class AdapterExperiment(BaseExperiment, ModelExperimentMixin):
    """
    Experiment class for fine-tuning with adapter layers
    Freezes base model parameters and trains only adapter module parameters
    """
    def __init__(self, config_path, experiment_type="adapter_finetuning"):
        super().__init__(config_path, experiment_type=experiment_type)
        ModelExperimentMixin.__init__(self, self.config)
        self.adapter_name = "toxicity_adapter"
        self.adapter_identifier = self.config.adapter_identifier

    def setup_adapter_model(
        self
    ):
        """
        Setup adapter configuration and prepare model for adapter training
        """

        adapters.init(self.model)

        self.model.add_adapter(
            self.adapter_name, 
            config=self.adapter_identifier
        )

        self.model.train_adapter(self.adapter_name)
    

    def train(
        self, 
        language, 
        use_class_weights=False
    ):
        """
        Modified training method to incorporate adapters
        
        Args:
            language (str): Language to train on
            use_class_weights (bool): Whether to apply class weights
        """
        logger.info(f"Starting adapter training for language: {language}")

        # Prepare class weights if needed
        class_weights = None
        if use_class_weights:
            class_weights = self.calculate_class_weights(self.datasets["train"].labels)

        # Apply adapters to the model
        self.setup_adapter_model()

        # Compute adapter trainable parameters percentage
        trainable_params, total_params = self.count_parameters()
        trainable_percentage = (trainable_params / total_params) * 100
        logger.info(f"Trainable Parameters: {trainable_params} ({trainable_percentage:.2f}%)")
        logger.info(f"Total Parameters: {total_params}")

        model_save_path = self.models_dir / f"{language}_adapter_model"

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

        train_results = trainer.train()

        trainer.save_model(str(model_save_path))

        metrics = train_results.metrics
        metrics.update({
            "total_params": total_params,
            "trainable_params": trainable_params,
            "trainable_percentage": trainable_percentage
        })

        self.save_json(
            train_results.metrics,
            self.models_dir / "training_metrics.json",
        )

        logger.info(f"Completed adapter training for language: {language}")


if __name__ == "__main__":
    config_path = "configs/adapter_config.yaml"
    languages = ["bodo"]
    experiment = AdapterExperiment(config_path)
    experiment.run_experiment(
        languages=languages, 
        use_class_weights=True
    )

