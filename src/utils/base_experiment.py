from datetime import datetime
from src.utils.helpers import save_json_with_numpy_conversion
from pathlib import Path
from src.project_setup import ProjectSetup
from sklearn.metrics import classification_report
from typing import Dict, Any
import logging, yaml


class BaseExperiment:
    """Base class for experiments with shared functionality"""
    def __init__(self, experiment_type: str):
        self.experiment_type = experiment_type
        self.config = {}
        self.setup_paths()
        self.setup_logging()
        
    def setup_paths(self):
        """Create necessary directories for results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = Path(ProjectSetup.RESULTS_DIR) / f"{self.experiment_type}/{timestamp}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        self.predictions_dir = self.experiment_dir / "predictions"
        self.metrics_dir = self.experiment_dir / "metrics"
        self.logs_dir = self.experiment_dir / "logs"
        self.models_dir = self.experiment_dir / "models"
        
        for dir_path in [self.predictions_dir, self.metrics_dir, self.logs_dir, self.models_dir]:
            dir_path.mkdir(exist_ok=True)
            
    def setup_logging(self):
        """Configure logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.logs_dir / "experiment.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def calculate_metrics(self, true_labels: list, predictions: list) -> dict:
        """Calculate classification metrics"""
        return classification_report(true_labels, predictions, output_dict=True, zero_division=0)
    
    def save_config(self):
        """Save experiment configuration"""
        with open(self.experiment_dir / "experiment_config.yaml", 'w') as f:
            yaml.dump(self.config, f)

    def save_metrics(self, metrics: Dict[str, Any], filename: str = "metrics.json"):
        """Save metrics to a file"""
        metrics_path = self.metrics_dir / filename
        save_json_with_numpy_conversion(metrics, metrics_path)

    def save_training_metrics(self, metrics: Dict[str, Any], filename: str = "training_metrics.json"):
        """Save training metrics to a file"""
        training_metrics_path = self.models_dir / filename
        save_json_with_numpy_conversion(metrics, training_metrics_path)

    def save_predictions(
            self,
            predictions: list,
            true_labels: list,
            save_path: Path,
            logits: list = None):
        """Save predictions and true labels to file"""
        results = {
            'predictions': predictions,
            'true_labels': true_labels
        }
        if logits:
            results['logits'] = logits
        save_json_with_numpy_conversion(results, save_path)
            
