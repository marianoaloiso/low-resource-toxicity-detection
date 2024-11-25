from datetime import datetime
from pathlib import Path
from src.project_setup import ProjectSetup
from src.utils.model import ModelConfig
from sklearn.metrics import classification_report
from typing import Dict, Any
import numpy as np
import json, logging, yaml


class BaseExperiment:
    """Base class for experiments with shared functionality"""
    def __init__(self, config_path: str, experiment_type: str):
        with open(config_path) as f:
            self.config = ModelConfig.from_dict(yaml.safe_load(f))
        self.experiment_type = experiment_type
        self.setup_paths()
        self.setup_logging()
        
    def _convert_to_json(self, o):
        if isinstance(o, np.int64): return int(o)
        if isinstance(o, np.float64): return float(o)
        raise TypeError
        
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
        
    def save_config(self):
        """Save experiment configuration"""
        with open(self.experiment_dir / "config.yaml", 'w') as f:
            yaml.dump(self.config, f)

    def calculate_metrics(self, true_labels: list, predictions: list) -> dict:
        """Calculate classification metrics"""
        return classification_report(true_labels, predictions, output_dict=True, zero_division=0)
    
    def save_metrics(self, metrics: Dict[str, Any], save_path: str):
        """Save metrics to a file"""
        with open(self.metrics_dir / save_path, 'w') as f:
            json.dump(metrics, f, default=self._convert_to_json)

    def save_predictions(self, predictions: list, true_labels: list, save_path: Path):
        """Save predictions and true labels to file"""
        results = {
            'predictions': predictions,
            'true_labels': true_labels
        }
        with open(save_path, 'w') as f:
            json.dump(results, f, default=self._convert_to_json)
            
