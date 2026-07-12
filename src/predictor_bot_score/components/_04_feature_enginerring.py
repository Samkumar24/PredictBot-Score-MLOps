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
from src.predictor_bot_score.utils.src_util_s3_ import *


class Feature_engineering:

    def __init__(self, config: FeatureEngineeringConfig):
        self.config = config
        self.BUCKET_NAME = self.config.bucket_name
        self.s3 =  s3_login()
        self.data   = self.read_data()
        self.pipeline_run_id = datetime.now().strftime("%Y_%m_%d_%H")

    def read_data(self):

        try:

            keys = []

            logger.info("=" * 50)
            logger.info("DATA VALIDATION PIPELINE STARTED")
            logger.info("=" * 50)
            
            for page in self.s3.list_objects_v2(Bucket=self.BUCKET_NAME,Prefix='data_validation')['Contents']:
                if page.get('Key').endswith('.csv'):
                    keys.append(page.get('Key'))

            logger.info("S3 . Connection exists ")

            combined_file_key = sorted(keys)[-1]


            obj = self.s3.get_object(Bucket=self.BUCKET_NAME,Key=combined_file_key)
            transformed_val_df = pd.read_csv(io.BytesIO(obj['Body'].read()))

            transformed_val_df["timestamp"] = pd.to_datetime(transformed_val_df["timestamp"], utc=True)

            transformed_val_df["bot_score"] = transformed_val_df["bot_score"].astype(float)
            
            return transformed_val_df
        
        except ClientError as e:
            raise
            
        except Exception as e:
            logger.error(f"Transformation failed: {e}")
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

            logger.info("=" * 50)
            logger.info("FEATURE ENGINEERING COMPLETE")
            logger.info("=" * 50)

        except Exception as e:
            logger.error(f"Feature engineering failed: {str(e)}")
            raise

        except Exception as e:
            logger.error(f"Failed to save featured data: {str(e)}")
            raise

    def _split_data(self):
        try:
            logger.info("STEP 5 - CHRONOLOGICAL TRAIN/VAL/TEST SPLIT")
            
            # 1. Your existing splitting logic
            total = len(self.data)
            splits = {
                "train": self.data.iloc[:int(total * 0.70)],
                "val":   self.data.iloc[int(total * 0.70):int(total * 0.80)],
                "test":  self.data.iloc[int(total * 0.80):]
            }

            return splits
        except Exception as e:
            raise e
        
    
    def run(self):
        try:
            self.build_features()
            splits    = self._split_data()
            run_folder = f"run__{self.pipeline_run_id}"

            for name, df in splits.items():
                if not isinstance(df, pd.DataFrame):
                    df = pd.DataFrame(df)

                # ── S3 paths — CSV and manifest are separate keys ──
                split_folder = f"{run_folder}/{name}__{self.pipeline_run_id}"
                csv_key      = f"feature_engineering/{split_folder}/{name}_data.csv"
                manifest_key = f"feature_engineering/{split_folder}/{name}_manifest.json"

                # ── local path — created before writing ────────────
                local_dir = os.path.join(
                    self.config.featured_data_dir,
                    run_folder,
                    name
                )
                os.makedirs(local_dir, exist_ok=True)   # ← create BEFORE writing

                # ── build manifest ─────────────────────────────────
                manifest = save_manifest(
                    pipline_id = self.pipeline_run_id,
                    output_key = csv_key
                )

                # ── save CSV to S3 ─────────────────────────────────
                save_file_s3(
                    df          = df,
                    output_key  = csv_key,
                    BUCKET_NAME = self.BUCKET_NAME,
                    s3_client   = self.s3
                )

                # ── save manifest to S3 (separate key) ────────────
                self_s3_mainfest(
                    manifest    = manifest,
                    output_key  = manifest_key,   # ← different key from CSV
                    BUCKET_NAME = self.BUCKET_NAME,
                    s3_client   = self.s3
                )

                # ── save CSV locally ───────────────────────────────
                df.to_csv(os.path.join(local_dir, f"{name}_data.csv"), index=False)

                # ── save manifest locally ──────────────────────────
                with open(os.path.join(local_dir, "manifest.json"), 'w') as f:
                    json.dump(manifest, f, indent=2)   # ← write directly, no nested subfolders

                logger.info(f"Processed {name}: {len(df)} rows")

            logger.info("=" * 50)
            logger.info("FEATURE ENGINEERING PIPELINE COMPLETED")
            logger.info("=" * 50)

        except Exception as e:
            logger.exception(f"Critical error in Feature Engineering: {e}")
            raise



        