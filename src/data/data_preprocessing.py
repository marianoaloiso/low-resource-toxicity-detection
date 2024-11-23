from src.project_setup import ProjectSetup
from sklearn.model_selection import train_test_split
import pandas as pd
import os


def load_data(language: str) -> pd.DataFrame:
    """
    Loads a CSV file for a specific language.

    Args:
        language (str): The name of the language.

    Returns:
        pd.DataFrame: The loaded dataframe.
    """
    data_path = f"{ProjectSetup.RAW_DATA_DIR}/{language}/{ProjectSetup.RAW_FILE}"
    try:
        df = pd.read_csv(data_path, index_col=0)
        df.index.name = None
        return df
    except FileNotFoundError:
        raise ValueError(f"Data file not found for language: {language}")
  
def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the data by handling missing values, duplicates, renaming columns, and label encoding.

    Args:
        df (pd.DataFrame): The input dataframe.

    Returns:
        pd.DataFrame: The preprocessed dataframe.
    """
    return (
        df.dropna()
        .drop_duplicates()
        .rename(columns={"task_1": "label"})
        .assign(label=lambda x: x["label"].map({"NOT": 0, "HOF": 1}))
    )

def split_data(language, test_size=0.1, val_size=0.1):
    """
    Splits data for a specific language into training, validation, and test sets.

    Args:
        language (str): The name of the language.
        test_size (float, optional): Proportion of the dataset for test split (default: 0.1).
        val_size (float, optional): Proportion of the dataset for validation split (default: 0.1).
        random_state (int, optional): Random seed for reproducibility.

    Returns:
        tuple: Three dataframes, (train_df, val_df, test_df).
    """
    df = load_data(language)
    df = process_data(df)
    random_state = ProjectSetup.RANDOM_STATE
    train_temp, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state,
        stratify=df['label']
    )
    train_df, val_df = train_test_split(
        train_temp, 
        test_size=val_size / (1 - test_size),
        random_state=random_state,
        stratify=train_temp['label']
    )
    return train_df, val_df, test_df

def main():
    for language in ProjectSetup.LANGUAGES:
        print(f"Processing language: {language}")
        train_df, val_df, test_df = split_data(language)

        output_dir = os.path.join(ProjectSetup.PROCESSED_DATA_DIR, language)
        os.makedirs(output_dir, exist_ok=True)

        train_df.to_csv(os.path.join(output_dir, ProjectSetup.TRAIN_FILE), index=False)
        val_df.to_csv(os.path.join(output_dir, ProjectSetup.VALIDATION_FILE), index=False)
        test_df.to_csv(os.path.join(output_dir, ProjectSetup.TEST_FILE), index=False)

        print(f"Splits for {language} saved successfully.")

if __name__ == "__main__":
    main()

