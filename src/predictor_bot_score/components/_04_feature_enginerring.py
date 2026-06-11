from dataclasses import dataclass
from pathlib import Path
from src.predictor_bot_score.config.configuration import yaml_load , create_directories
from src.predictor_bot_score.logger import logger
from src.predictor_bot_score.constants import CONFIG_PATH
import pandas as pd
from datetime import datetime
import glob
import os
from src.predictor_bot_score.config.configuration import FeatureEngineeringConfig


class Feature_engineering:

    def __init__(self, config: FeatureEngineeringConfig):
        self.config = config
        self.data   = self.read_data()

    def read_data(self):
        try:
            logger.info(f"Attempting to read data from: {self.config.transformed_data_path}")

            files = glob.glob(
                os.path.join(self.config.transformed_data_path, "transformed*.csv")
            )

            if not files:
                raise FileNotFoundError(
                    f"No transformed files found in {self.config.transformed_data_path}"
                )

            latest_file = max(files, key=os.path.getmtime)
            logger.info(f"Latest file found: {latest_file}")

            data = pd.read_csv(latest_file)
            data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
            data["bot_score"]  = data["bot_score"].astype(float)

            logger.info(f"Data successfully loaded - {len(data)} rows")
            return data

        except FileNotFoundError:
            print("--- ALERT: No transformed file found. Run transformation first. ---")
            logger.error(f"No transformed file found at: {self.config.transformed_data_path}")
            raise

        except Exception as e:
            logger.error(f"Failed to read data: {str(e)}")
            raise

    def build_features(self):
        try:
            logger.info("=" * 50)
            logger.info("FEATURE ENGINEERING PIPELINE STARTED")

            
            logger.info("STEP 1 - LAG FEATURES")
            logger.info("-" * 50)
            for col_name, shift_value in self.config.lag_columns.items():
                self.data[col_name] = self.data["bot_score"].shift(shift_value)
                logger.info(f"Created: {col_name} (shift={shift_value})")
            logger.info("PASSED - Lag features created")

            logger.info("")
            logger.info("STEP 2 - ROLLING STD FEATURES")
            logger.info("-" * 50)
            for col_name, window in self.config.rolling.items():
                self.data[col_name] = self.data["bot_score"].rolling(window).std()
                logger.info(f"Created: {col_name} (window={window})")
            logger.info("PASSED - Rolling features created")

            
            logger.info("STEP 3 - DROP NaN ROWS")
            logger.info("-" * 50)
            before = len(self.data)
            self.data = self.data.dropna(
                subset=self.config.features
            ).reset_index(drop=True)
            after = len(self.data)
            logger.info(f"Dropped {before - after} NaN rows from lag/rolling")
            logger.info(f"PASSED - {after} rows remaining")

            
            logger.info("STEP 4 - SELECT FINAL COLUMNS")
            logger.info("-" * 50)
            final_cols  = self.config.features + [self.config.target_column]
            self.data   = self.data[final_cols]
            logger.info(f"Final columns: {final_cols}")
            logger.info("PASSED - Final columns selected")

            self._save_data()
            self._split_data()

            logger.info("=" * 50)
            logger.info("FEATURE ENGINEERING COMPLETE")
            logger.info("=" * 50)

        except Exception as e:
            logger.error(f"Feature engineering failed: {str(e)}")
            raise

    def _save_data(self):
        try:
            os.makedirs(self.config.featured_data_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
            out_path  = os.path.join(
                self.config.featured_raw_data,
                f"featured_{timestamp}.csv"
            )
            self.data.to_csv(out_path, index=False)
            logger.info(f"Data saved - {out_path}")

        except Exception as e:
            logger.error(f"Failed to save featured data: {str(e)}")
            raise

    def _split_data(self):
        try:
            
            
            logger.info("STEP 5 - CHRONOLOGICAL TRAIN/VAL/TEST SPLIT")
            logger.info("-" * 50)

            total     = len(self.data)
            train_end = int(total * 0.70)
            val_end   = int(total * 0.80)

            train = self.data.iloc[:train_end]
            val   = self.data.iloc[train_end:val_end]
            test  = self.data.iloc[val_end:]

            logger.info(f"Total rows : {total}")
            logger.info(f"Train  70% : {len(train)} rows")
            logger.info(f"Val    10% : {len(val)} rows")
            logger.info(f"Test   20% : {len(test)} rows") 

            version = datetime.now().strftime("%Y_%m_%d_%H_%M")

            splits = {
                "train" : (train, self.config.train_data),
                "val"   : (val,   self.config.val_data),
                "test"  : (test,  self.config.test_data),
            }

            for split_name , (split_df ,split_dir) in splits.items():
                file_name = f"{split_name}_{version}.csv"
                output_path = os.path.join(split_dir,file_name)
                split_df.to_csv(output_path, index=False)
            logger.info(f"Saved {split_name} --- {output_path}")

            logger.info(f"Version : {version}")
            logger.info("PASSED - Train/Val/Test split complete")

        except Exception as e:
            logger.error(f"Failed to save featured data: {str(e)}")
            raise