from src.data.data_loader import DataLoader
from src.utils.base_experiment import BaseExperiment
from src.utils.model import CustomNLIPipeline
from transformers import pipeline
import logging

logger = logging.getLogger(__name__)

class ZeroShotInferenceExperiment(BaseExperiment):
    """Zero-shot inference experiment using NLI"""

    def __init__(self, config_path, experiment_type):
        super().__init__(config_path, experiment_type)
        self.model_name = self.config.model_name

        if self.model_name == "ai4bharat/indic-bert":
            self.pipeline = CustomNLIPipeline(model_name=self.model_name)
        elif self.model_name == "facebook/bart-large-mnli":
            self.pipeline = pipeline("zero-shot-classification", model=self.model_name)
        else:
            raise ValueError(f"Model {self.model_name} not supported for zero-shot inference")

    def evaluate(
            self,
            language,
            hypothesis_template="This text is {}.",
            candidate_labels={1: "toxic", 0: "non-toxic"},
            dataset_split="test"
        ):
        """Evaluate the model using the custom zero-shot pipeline"""
        logger.info(f"Evaluating on {dataset_split} set for language: {language}")

        dataset = self.datasets[dataset_split]
        predictions = []
        true_labels = []
        candidate_labels_text = list(candidate_labels.keys())

        for text, label in zip(dataset.texts, dataset.labels):
            true_labels.append(label)

            results = self.pipeline.classify(
                text,
                candidate_labels_text,
                hypothesis_template
            )

            # Get the highest scoring label
            predicted_label = max(results, key=results.get)
            predictions.append(candidate_labels[predicted_label])

        # Save predictions
        self.save_predictions(
            predictions=predictions,
            true_labels=true_labels,
            save_path=self.predictions_dir / f"{language}_predictions_{dataset_split}.json"
        )

        # Compute metrics
        metrics = self.calculate_metrics(true_labels, predictions)
        return metrics

    def run_experiment(self, target_languages, hypothesis_template=None, candidate_labels=None):
        """Run the NLI experiment"""

        for language in target_languages:
            logger.info(f"Starting experiment for language: {language}")

            data_loader = DataLoader(tokenizer=None, max_length=self.config.max_length)
            data_loader.tokenizer = self.pipeline.tokenizer
            self.datasets = data_loader.load_language_data(language)

            train_metrics = self.evaluate(
                language,
                dataset_split="train",
                hypothesis_template=hypothesis_template,
                candidate_labels=candidate_labels
            )
            val_metrics = self.evaluate(
                language,
                dataset_split="validation",
                hypothesis_template=hypothesis_template,
                candidate_labels=candidate_labels
            )
            test_metrics = self.evaluate(
                language,
                dataset_split="test",
                hypothesis_template=hypothesis_template,
                candidate_labels=candidate_labels
            )

            all_metrics = {"train": train_metrics, "validation": val_metrics, "test": test_metrics}
            self.save_metrics(all_metrics, save_path=self.metrics_dir / f"{language}_metrics.json")

            logger.info(f"Completed experiment for language: {language}")

if __name__ == "__main__":
    
    candidate_labels_bodo = {"खराब बाथा": 1, "सद्भावपूर्ण बाथा": 0}
    candidate_labels_english = {"toxic": 1, "non-toxic": 1}

    hypothesis_template_bodo = "एया बाथा {} हो।"
    hypothesis_template_english = "This text is {}."

    config_path = "configs/zero_shot_inference_config.yaml"
    target_languages = ["bodo"]

    # English zero-shot inference
    experiment = ZeroShotInferenceExperiment(
        config_path,
        experiment_type="zero_shot_inference/english"
    )
    experiment.run_experiment(
        target_languages=target_languages,
        hypothesis_template=hypothesis_template_english,
        candidate_labels=candidate_labels_english
    )

    # Bodo zero-shot inference
    experiment = ZeroShotInferenceExperiment(
        config_path,
        experiment_type="zero_shot_inference/bodo"
    )
    experiment.run_experiment(
        target_languages=target_languages,
        hypothesis_template=hypothesis_template_bodo,
        candidate_labels=candidate_labels_bodo,
    )

