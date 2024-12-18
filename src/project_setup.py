from pathlib import Path

class ProjectSetup:
    # Data directories
    ROOT_DIR = Path(__file__).parent.parent.absolute()
    DATA_DIR = ROOT_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"

    # Data files
    RAW_FILE = "train.csv" # Original name of the file. Test set did not have labels.
    TRAIN_FILE = "train.csv"
    VALIDATION_FILE = "validation.csv"
    TEST_FILE = "test.csv"
    BALANCED_TRAIN_FILE = "balanced_train.csv"
    
    # Results directory
    RESULTS_DIR = ROOT_DIR / "results"

    # Dummy directory created during training
    DUMMY_DIR = ROOT_DIR / "results" / "tmp"
    
    # Languages
    LANGUAGES = ["assamese", "bengali", "bodo"]

    # Labels
    LABELS = {0: "not-toxic", 1: "toxic"}
    
    # Model parameters
    RANDOM_STATE = 819

    # Create processed data directory if it does not exist
    def create_processed_data_dir(language: str):
        path = ProjectSetup.PROCESSED_DATA_DIR / language
        path.mkdir(parents=True, exist_ok=True)

    def get_train_path(language: str):
        return ProjectSetup.PROCESSED_DATA_DIR / language / ProjectSetup.TRAIN_FILE

    def get_validation_path(language: str):
        return ProjectSetup.PROCESSED_DATA_DIR / language / ProjectSetup.VALIDATION_FILE
    
    def get_test_path(language: str):
        return ProjectSetup.PROCESSED_DATA_DIR / language / ProjectSetup.TEST_FILE

    def get_balanced_train_path(language: str):
        return ProjectSetup.PROCESSED_DATA_DIR / language / ProjectSetup.BALANCED_TRAIN_FILE
    
