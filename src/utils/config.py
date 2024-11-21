import os


class Config:
    # Constants
    LANGUAGES = ["assamese", "bengali", "bodo"]

    # Data directories
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    DATA_DIR = os.path.join(ROOT_DIR, "data")
    RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

    # File names
    RAW_FILE = "train.csv" # Original name of the file. Test set did not have labels.
    TRAIN_FILE = "train.tsv"
    VAL_FILE = "dev.tsv"
    TEST_FILE = "test.tsv"

    # Model parameters
    RANDOM_STATE = 819

