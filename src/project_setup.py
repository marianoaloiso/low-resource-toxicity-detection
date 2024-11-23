import os

class ProjectSetup:
    # Data directories
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
    DATA_DIR = os.path.join(ROOT_DIR, "data")
    RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

    # Data files
    RAW_FILE = "train.csv" # Original name of the file. Test set did not have labels.
    TRAIN_FILE = "train.csv"
    VALIDATION_FILE = "dev.csv"
    TEST_FILE = "test.csv"

    # Results directory
    RESULTS_DIR = os.path.join(ROOT_DIR, "results")

    # Languages
    LANGUAGES = ["assamese", "bengali", "bodo"]
    
    # Model parameters
    RANDOM_STATE = 819

