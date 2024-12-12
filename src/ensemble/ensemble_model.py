from src.data.data_loader import DataLoader as CustomDataLoader
from src.utils.base_experiment import BaseExperiment
from src.utils.model import ModelExperimentMixin, WeightedTrainer
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict
import torch.optim as optim
import logging, os, torch

logger = logging.getLogger(__name__)


class LogitsEnsembleNeuralNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        """
        Two-layer feed-forward neural network for logits ensemble

        Args:
            input_size (int): Total number of input logits from different experiments
            hidden_size (int): Size of the hidden layer
        """
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return torch.sigmoid(self.network(x))
    

class EnsembleModelExperiment(BaseExperiment, ModelExperimentMixin):
    """Fine-tuning model on individual datasets experiment"""

    def __init__(self, config_path: str):
        super().__init__(experiment_type="ensemble_model")
        ModelExperimentMixin.__init__(self, config_path)
    
        self.base_results_dir = "results/oversampled_dataset_results"
        self.experiment_paths = {
            "Individual Language Fine-tuning": "monolingual_finetuning",
            "Cross-Lingual Transfer Learning": "crosslingual_transfer",
            "Multilingual Fine-Tuning": "multilingual_finetuning",
            "Adapters": "adapter_finetuning",
            "LoRA": "lora_finetuning",
            "Backtranslation": "backtranslation",
            "Transliteration": "transliteration/bengali",
            "Noise Injection": "noise_injection/noise_level=0.2"
        }
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def find_latest_dir(path: str) -> str:
        """Find the latest directory in the given path"""
        return sorted(os.listdir(path))[-1]
    
    def load_model_and_tokenizer(self, path: str):
        """Load tokenizer and model from the specified path."""
        tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-bert")
        model = AutoModelForSequenceClassification.from_pretrained(path, num_labels=2)
        return tokenizer, model
    
    def prepare_ensemble_input(self, predictions: Dict[str, Dict]):
        """Aggregate logits from all experiments to prepare ensemble input."""
        logits_list, ground_truth = [], None

        for result in predictions.values():
            if "error" not in result:
                logits_list.append(result["logits"])
                if ground_truth is None:
                    ground_truth = result["true_labels"]

        ensemble_input = torch.cat(logits_list, dim=-1)
        return ensemble_input, ground_truth

    def get_logits_experiment(
            self,
            experiment: str,
            language: str
        ):
        """Load model and tokenizer from the latest experiment and get logits

        Args:
            experiment (str): Experiment name
            language (str): Language for which to get logits
        
        Returns:
            dict: Logits and predicted labels
        """
        experiment_path = self.experiment_paths[experiment]
        latest_timestamp = self.find_latest_dir(f"{self.base_results_dir}/{experiment_path}")

        full_path_model = f"{self.base_results_dir}/{experiment_path}/{latest_timestamp}/models"

        tokenizer, model = self.load_model_and_tokenizer(full_path_model)
        model.eval()

        loader = CustomDataLoader(
            tokenizer=tokenizer,
            max_length=self.config.max_length,
        )

        dataset = loader.load_language_data(language)

        trainer = WeightedTrainer(model=model)

        predictions = trainer.predict(dataset["train"])
        logits = predictions.predictions
        pred_labels = logits.argmax(-1)
        
        return {
            "logits": torch.tensor(logits).float(),
            "pred_labels": torch.tensor(pred_labels),
            "true_labels": torch.tensor(dataset["train"].labels),
        }
    
    def train_ensemble_model(self, ensemble_input: torch.Tensor, ground_truth: torch.Tensor) -> nn.Module:
        """Train the ensemble neural network."""
        dataset = TensorDataset(ensemble_input, ground_truth)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )

        ensemble_model = LogitsEnsembleNeuralNetwork(
            input_size=ensemble_input.size(1),
            hidden_size=self.config.hidden_size
        ).to(self.device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(ensemble_model.parameters(), lr=0.001)

        num_epochs = self.config.num_epochs
        for epoch in range(num_epochs):
            ensemble_model.train()
            total_loss = 0
            for batch_input, batch_labels in dataloader:
                optimizer.zero_grad()
                
                outputs = ensemble_model(batch_input)
                loss = criterion(outputs.squeeze(), batch_labels.float())
                
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}')
        
        return ensemble_model
    
    def evaluate_ensemble_model(
            self,
            ensemble_model: nn.Module,
            ensemble_input: torch.Tensor
        ) -> float:
        """Evaluate the ensemble model on the training data."""
        ensemble_model.eval()
        with torch.no_grad():
            outputs = ensemble_model(ensemble_input.to(self.device))
            predicted_labels = (outputs > 0.5)
        return predicted_labels
    
    def run_experiment(self, language: str):
        """Run the full ensemble experiment pipeline."""
        predictions = {}
        for experiment in self.experiment_paths:
            predictions[experiment] = self.get_logits_experiment(experiment, language)
        
        ensemble_input, ground_truth = self.prepare_ensemble_input(predictions)

        logger.info(f"Ensemble input shape: {ensemble_input.size()}")
        logger.info(f"Ground truth shape: {ground_truth.size()}")

        ensemble_model = self.train_ensemble_model(ensemble_input, ground_truth)

        predicted_labels = self.evaluate_ensemble_model(ensemble_model, ensemble_input)

        metrics = self.calculate_metrics(
            true_labels=ground_truth.tolist(),
            predictions=predicted_labels.tolist()
        )

        self.save_metrics(metrics, f"{language}_metrics.json")

        self.save_config()

if __name__ == "__main__":
    config_path = "configs/ensemble_model_config.yaml"
    language = "bodo"
    experiment = EnsembleModelExperiment(config_path)
    experiment.run_experiment(language)

