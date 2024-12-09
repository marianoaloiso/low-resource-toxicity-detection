from src.data.data_loader import DataLoader
from src.project_setup import ProjectSetup
from src.utils.base_experiment import BaseExperiment
from src.utils.model import ModelExperimentMixin
import logging, random


logger = logging.getLogger(__name__)

class NoiseInjectionExperiment(BaseExperiment, ModelExperimentMixin):
    """
    Experiment to augment training data with controlled noise
    Supports various noise injection strategies for data augmentation
    """

    def __init__(self, config_path, augmentation_factor=1.5, noise_level=0.1):
        """
        Initialize the experiment with noise configuration
        
        Args:
            config_path (str): Path to configuration file
            augmentation_factor (float): Factor to multiply training data size
        """
        super().__init__(experiment_type=f"noise_injection/noise_level={noise_level}")
        ModelExperimentMixin.__init__(self, config_path)

        self.noise_types = ['word_swap', 'word_drop', 'word_replace']
        self.noise_level = noise_level
        self.augmentation_factor = augmentation_factor

    def inject_noise(self, text: str) -> str:
        """
        Inject noise into a given text with a specified noise level.

        Args:
            text (str): Input text to add noise to.

        Returns:
            str: Noisy text.
        """

        words = text.split()

        if not words or len(words) == 1:
            return text

        noisy_words = []
        for word in words:
            if random.random() < self.noise_level:
                noise_type = random.choice(self.noise_types)

                if noise_type == 'word_swap':
                    swap_index = random.randint(0, len(words) - 1)
                    noisy_words.append(words[swap_index])
                elif noise_type == 'word_drop':
                    noisy_words.append('')
                elif noise_type == 'word_replace':
                    noisy_words.append(random.choice(words))
            else:
                noisy_words.append(word)

        return ' '.join(noisy_words)

    def augment_training_data(self, dataset):
        """
        Augment training dataset by injecting noise
        
        Args:
            dataset: Training dataset to augment
        
        Returns:
            Augmented dataset
        """
        # Calculate number of augmented samples
        original_texts = dataset.texts
        original_labels = dataset.labels
        
        # Create augmented samples
        noisy_texts = []
        noisy_labels = []
        
        # Determine number of augmented samples
        num_augmented_samples = int(len(original_texts) * (self.augmentation_factor - 1))

        for _ in range(num_augmented_samples):
            # Randomly select a sample to augment
            idx = random.randint(0, len(original_texts) - 1)
            noisy_text = self.inject_noise(original_texts[idx])
            
            noisy_texts.append(noisy_text)
            noisy_labels.append(original_labels[idx])
        
        dataset.texts = original_texts + noisy_texts
        dataset.labels = original_labels + noisy_labels
        
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
        loader = DataLoader(
            tokenizer=None,
            max_length=self.config.max_length,
        )

        for language in languages:
            logger.info(f"Starting noise injection experiment for language: {language}")

            # Initialize model for each language
            model, tokenizer = self.load_automodel(
                self.model_name, 
                self.num_labels
            )
            loader.tokenizer = tokenizer

            # Load language data
            datasets = loader.load_language_data(language)

            # Augment only training data
            datasets['train'] = self.augment_training_data(datasets['train'])

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
            
            logger.info(f"Completed noise injection experiment for language: {language}")


if __name__ == "__main__":
    augmentation_factor = 1.5
    noise_levels = [0.1, 0.2, 0.3]
    config_path = "configs/monolingual_finetuning_config.yaml"
    
    for noise_level in noise_levels:
        experiment = NoiseInjectionExperiment(
            config_path,
            augmentation_factor=augmentation_factor,
            noise_level=noise_level
        )
        experiment.run_experiment(languages=['bodo'])

