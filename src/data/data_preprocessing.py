from src.project_setup import ProjectSetup
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import pandas as pd


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
    train_temp, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=ProjectSetup.RANDOM_STATE,
        stratify=df['label']
    )
    train_df, val_df = train_test_split(
        train_temp, 
        test_size=val_size / (1 - test_size),
        random_state=ProjectSetup.RANDOM_STATE,
        stratify=train_temp['label']
    )
    return train_df, val_df, test_df

def balance_train_dataset(train_df, oversample=False, undersample=False):
    """
    Balances the training dataset using oversampling or undersampling.

    Args:
        train_df (pd.DataFrame): The training dataframe containing at least 'text' and 'label' columns.
        oversample (bool, optional): If True, performs oversampling to balance the dataset (default: False).
        undersample (bool, optional): If True, performs undersampling to balance the dataset (default: False).

    Returns:
        pd.DataFrame: A balanced training dataframe.
    """
    if not (oversample or undersample):
        raise ValueError("Either oversample or undersample must be True.")

    classes = train_df['label'].unique()
    max_count = train_df['label'].value_counts().max()
    min_count = train_df['label'].value_counts().min()

    balanced_dfs = []

    for label in classes:
        class_df = train_df[train_df['label'] == label]
        if oversample:
            if len(class_df) == max_count:
                resampled_df = class_df
            else:
                resampled_df = resample(
                    class_df,
                    replace=True,
                    n_samples=max_count,
                    random_state=ProjectSetup.RANDOM_STATE
                )
        elif undersample:
            resampled_df = resample(
                class_df,
                replace=False,
                n_samples=min_count,
                random_state=ProjectSetup.RANDOM_STATE
            )
        balanced_dfs.append(resampled_df)

    balanced_train_df = pd.concat(balanced_dfs)
    return balanced_train_df.sample(frac=1, random_state=ProjectSetup.RANDOM_STATE).reset_index(drop=True)

def main():
    for language in ProjectSetup.LANGUAGES:
        print(f"Processing language: {language}")
        train_df, val_df, test_df = split_data(language)

        balanced_train_df = balance_train_dataset(train_df, oversample=True)

        ProjectSetup.create_processed_data_dir(language)

        train_df.to_csv(ProjectSetup.get_train_path(language), index=False)
        val_df.to_csv(ProjectSetup.get_validation_path(language), index=False)
        test_df.to_csv(ProjectSetup.get_test_path(language), index=False)

        balanced_train_df.to_csv(ProjectSetup.get_balanced_train_path(language), index=False)

        print(f"Splits for {language} saved successfully.")

if __name__ == "__main__":
    main()

