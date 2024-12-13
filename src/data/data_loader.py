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
        max_length: int = 512,
        device: torch.device = torch.device("cpu")
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device

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
            "input_ids": encoding["input_ids"].squeeze(0).to(self.device),
            "attention_mask": encoding["attention_mask"].squeeze(0).to(self.device),
            "labels": torch.tensor(label, dtype=torch.long).to(self.device)
        }


class DataLoader:
    """Handles loading of data for different languages

    Args:
        tokenizer (PreTrainedTokenizer): Tokenizer to use for encoding text.
        max_length (int): Maximum length of input sequences.

    Attributes:
        tokenizer (PreTrainedTokenizer): Tokenizer to use for encoding text.
        max_length (int): Maximum length of input sequences.
    """
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer = None,
        max_length: int = 512
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def load_language_data(
            self,
            language: str,
            load_train: bool = True,
            load_validation: bool = True,
            load_test: bool = True,
            device: torch.device = torch.device("cpu")
        ) -> pd.DataFrame:
        """Load data for a specific language

        Args:
            language (str): Language for which to load data.
            load_train (bool): Load training data.
            load_validation (bool): Load validation data.
            load_test (bool): Load test data.
            device (torch.device): Device to use for loading data.
        
        Returns:
            Dict[str, pd.DataFrame]: Data for the specified language.
        """
        data = {}

        if load_train:
            train_path = ProjectSetup.get_train_path(language)
            assert train_path.exists(), f"Train file for language {language} not found"
            train_data = pd.read_csv(train_path)
            data["train"] = ToxicCommentsDataset(
                texts=train_data["text"].tolist(),
                labels=train_data["label"].tolist(),
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                device=device
            )

        if load_validation:
            validation_path = ProjectSetup.get_validation_path(language)
            assert validation_path.exists(), f"Validation file for language {language} not found"
            validation_data = pd.read_csv(validation_path)
            data["validation"] = ToxicCommentsDataset(
                texts=validation_data["text"].tolist(),
                labels=validation_data["label"].tolist(),
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                device=device
            )
        
        if load_test:
            test_path = ProjectSetup.get_test_path(language)
            assert test_path.exists(), f"Test file for language {language} not found"
            test_data = pd.read_csv(test_path)
            data["test"] = ToxicCommentsDataset(
                texts=test_data["text"].tolist(),
                labels=test_data["label"].tolist(),
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                device=device
            )

        logger.info(f"Loaded data for language: {language}")
        return data

    def load_combined_language_data(
        self,
        languages: List[str] = ProjectSetup.LANGUAGES,
        splits: List[str] = ["train", "validation", "test"],
        device: torch.device = torch.device("cpu")
    ) -> Dict[str, "ToxicCommentsDataset"]:
        """
        Load and combine data for multiple languages
        
        Args:
            languages (List[str]): List of languages to combine.
            splits (List[str]): Dataset splits to combine (e.g., train, validation, test).

        Returns:
            Dict[str, ToxicCommentsDataset]: Combined datasets for the specified splits.
        """
        combined_data = {split: {"texts": [], "labels": []} for split in splits}

        for language in languages:
            language_data = self.load_language_data(language)

            for split in splits:
                if split in language_data:
                    combined_data[split]["texts"].extend(language_data[split].texts)
                    combined_data[split]["labels"].extend(language_data[split].labels)

        for split in splits:
            combined_data[split] = ToxicCommentsDataset(
                texts=combined_data[split]["texts"],
                labels=combined_data[split]["labels"],
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                device=device
            )

        logger.info(f"Combined multilingual data for languages: {languages}")
        return combined_data
    
    