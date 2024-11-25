from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from typing import List, Dict
from src.project_setup import ProjectSetup
import pandas as pd
import logging, torch

logger = logging.getLogger(__name__)

class ToxicCommentsDataset(Dataset):
    """Dataset class for toxic comments classification"""
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


class DataLoader:
    """Handles loading of data for different languages"""
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def load_language_data(
            self,
            language: str,
            load_train: bool = True,
            load_validation: bool = True,
            load_test: bool = True
        ) -> pd.DataFrame:
        """Load data for a specific language"""
        data = {}

        if load_train:
            train_path = ProjectSetup.get_train_path(language)
            assert train_path.exists(), f"Train file for language {language} not found"
            train_data = pd.read_csv(train_path)
            data["train"] = ToxicCommentsDataset(
                texts=train_data["text"].tolist(),
                labels=train_data["label"].tolist(),
                tokenizer=self.tokenizer,
                max_length=self.max_length
            )

        if load_validation:
            validation_path = ProjectSetup.get_validation_path(language)
            assert validation_path.exists(), f"Validation file for language {language} not found"
            validation_data = pd.read_csv(validation_path)
            data["validation"] = ToxicCommentsDataset(
                texts=validation_data["text"].tolist(),
                labels=validation_data["label"].tolist(),
                tokenizer=self.tokenizer,
                max_length=self.max_length
            )
        
        if load_test:
            test_path = ProjectSetup.get_test_path(language)
            assert test_path.exists(), f"Test file for language {language} not found"
            test_data = pd.read_csv(test_path)
            data["test"] = ToxicCommentsDataset(
                texts=test_data["text"].tolist(),
                labels=test_data["label"].tolist(),
                tokenizer=self.tokenizer,
                max_length=self.max_length
            )

        logger.info(f"Loaded data for language: {language}")
        return data

    def load_all_languages(
        self,
        languages: List[str] = ProjectSetup.LANGUAGES
    ) -> Dict[str, pd.DataFrame]:
        """Load data for all languages"""
        datasets_dict = {}
        for language in languages:
            datasets_dict[language] = self.load_language_data(language)

        return datasets_dict
    
    