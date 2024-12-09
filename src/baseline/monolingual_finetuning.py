from src.utils.base_experiment import BaseExperiment
from src.utils.model import ModelExperimentMixin
import logging

logger = logging.getLogger(__name__)

class MonolingualFinetuningExperiment(BaseExperiment, ModelExperimentMixin):
    """Fine-tuning model on individual datasets experiment"""

    def __init__(self, config_path):
        super().__init__("monolingual_finetuning")
        ModelExperimentMixin.__init__(self, config_path)

if __name__ == "__main__":
    config_path = "configs/monolingual_finetuning_config.yaml"
    languages = ["bodo"]
    experiment = MonolingualFinetuningExperiment(config_path)
    experiment.run_experiment(languages=languages)

