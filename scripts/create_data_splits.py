import os
from src.data.data_preprocessing import split_data
from src.utils.config import Config

def main():
    for language in Config.LANGUAGES:
        print(f"Processing language: {language}")
        train_df, val_df, test_df = split_data(language)

        output_dir = os.path.join(Config.PROCESSED_DATA_DIR, language)
        os.makedirs(output_dir, exist_ok=True)

        train_df.to_csv(os.path.join(output_dir, Config.TRAIN_FILE), index=False, sep="\t")
        val_df.to_csv(os.path.join(output_dir, Config.VAL_FILE), index=False, sep="\t")
        test_df.to_csv(os.path.join(output_dir, Config.TEST_FILE), index=False, sep="\t")

        print(f"Splits for {language} saved successfully.")

if __name__ == "__main__":
    main()
    
