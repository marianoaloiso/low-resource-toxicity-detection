from indic_transliteration import sanscript
from src.data.data_loader import DataLoader
from src.utils.base_experiment import BaseExperiment
from src.utils.model import ModelExperimentMixin
import logging


logger = logging.getLogger(__name__)

class TransliterationFinetuningExperiment(BaseExperiment, ModelExperimentMixin):
    """
    Experiment to test XLM-R performance with transliterated datasets
    Supports transliteration between different Indic scripts and Latin script
    """

    def __init__(self, config_path, transliteration_target='bengali'):
        """
        Initialize the experiment with transliteration configuration
        
        Args:
            config_path (str): Path to configuration file
            transliteration_target (str): Target script for transliteration
        """
        super().__init__(experiment_type=f"transliteration/{transliteration_target}")
        ModelExperimentMixin.__init__(self, config_path)

        self.language_scripts = {
            'bengali': sanscript.BENGALI,
            'bodo': sanscript.DEVANAGARI,
            'assamese': sanscript.BENGALI,
            'latin': sanscript.ITRANS
        }
        self.transliteration_target = self.language_scripts.get(transliteration_target.lower())

    def transliterate_text(self, text, source_script):
        """
        Transliterate text from source script to target script
        
        Args:
            text (str): Input text to transliterate
            source_script (str): Source script of the text
        
        Returns:
            str: Transliterated text
        """
        try:
            return sanscript.transliterate(
                text, 
                source_script,
                self.transliteration_target
            )
        except Exception as e:
            logger.warning(f"Transliteration failed: {e}")
            return text

    def transliterate_dataset(self, dataset, language):
        """
        Augment dataset with transliterated texts (in-place)
        
        Args:
            dataset: Dataset to transliterate
            language (str): Source language
        
        Returns:
            Dataset: Transliterated dataset
        """
        source_script = self.language_scripts.get(language.lower())
        
        if not source_script:
            logger.warning(f"No script found for language {language}")
            return dataset
        
        transliterated_texts = [
            self.transliterate_text(text, source_script)
            for text in dataset.texts
        ]

        dataset.texts.extend(transliterated_texts)
        dataset.labels.extend(dataset.labels)

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
            logger.info(f"Starting experiment for language: {language}")

            # Initialize fresh model for each language
            model, tokenizer = self.load_automodel(self.model_name, self.num_labels)
            data_loader.tokenizer = tokenizer

            # Setup data for the language
            datasets = data_loader.load_language_data(language)

            # Augment training data with transliteration
            datasets['train'] = self.transliterate_dataset(datasets['train'], language)

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
            
            logger.info(f"Completed transliteration experiment for language: {language}")
        
        # Save configuration for the experiment
        self.save_config()


if __name__ == "__main__":
    language_pairs = [
        ('bodo', 'bengali'),
        ('bodo', 'latin')
    ]
    config_path = "configs/monolingual_finetuning_config.yaml"
    for source, target in language_pairs:
        experiment = TransliterationFinetuningExperiment(config_path, transliteration_target=target)
        experiment.run_experiment(languages=[source])

