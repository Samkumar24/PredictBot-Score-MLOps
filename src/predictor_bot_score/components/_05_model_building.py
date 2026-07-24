from dataclasses import dataclass
from pathlib import Path
from src.predictor_bot_score.config.configuration import yaml_load , create_directories
from src.predictor_bot_score.logger import logger
from src.predictor_bot_score.constants import CONFIG_PATH
from src.predictor_bot_score.utils.model_factory import get_model , get_fit_kwargs , get_mlflow_logger , MODEL_REGISTRY
from sklearn.metrics import mean_absolute_error
import os
import numpy as np
import pandas as pd
import glob
import mlflow
import lightgbm
import pickle
import json
from datetime import datetime
import sqlite3
import gc
from src.predictor_bot_score.entity import ModelTrainingConfig
from src.predictor_bot_score.utils.src_util_s3_ import *





class Model_Building :

    def __init__(self, config : ModelTrainingConfig):
        self.config      = config
        self.s3          = s3_login()
        self.BUCKET_NAME = self.config.bucket_name
        self.batch_id    = datetime.now().strftime("%Y_%m_%d_%H")
        self.train, self.val = self._read_data()

    def _read_data(self):
        try:
            logger.info("=" * 50)
            logger.info("Reading train / val splits from S3")
            logger.info("=" * 50)

            train_key = self._get_latest_key("feature_engineering", "train")
            val_key   = self._get_latest_key("feature_engineering", "val")

            train = self._read_csv_from_s3(key=train_key)
            val   = self._read_csv_from_s3(key=val_key)

            logger.info(f"Train : {len(train)} rows")
            logger.info(f"Val   : {len(val)} rows")
            return train, val

        except Exception as e:
            logger.error(f"Failed to read splits from S3: {e}")
            raise
    
    def _get_latest_key(self, prefix, split_name):
        """Find the latest CSV key for train/val/test under a prefix."""
        try:
            paginator = self.s3.get_paginator('list_objects_v2')
            keys = []
            for page in paginator.paginate(Bucket=self.BUCKET_NAME, Prefix=prefix):
                for obj in page.get('Contents', []):
                    key = obj['Key']
                    if split_name in key and key.endswith('.csv') :
                        keys.append(key)

            if not keys:
                raise FileNotFoundError(
                    f"No {split_name} CSV found under s3://{self.BUCKET_NAME}/{prefix}"
                )

            latest = sorted(keys)[-1]
            logger.info(f"Found {split_name} file: {latest}")
            return latest

        except ClientError as e:
            logger.error(f"S3 error listing {prefix}: {e.response['Error']['Message']}")
            raise

    def _read_csv_from_s3(self, key):
        try:
            obj = self.s3.get_object(Bucket=self.BUCKET_NAME, Key=key)
            df  = pd.read_csv(io.BytesIO(obj['Body'].read()))
            return df
        except ClientError as e:
            logger.error(f"Failed to read {key}: {e.response['Error']['Message']}")
            raise

    def prepare_data(self ,df :pd.DataFrame):
        try:
            X = df[self.config.features]
            y = df[self.config.target_column]

            return X ,y
        except Exception as e:
            raise e 
        
    def _smape(self, actual , predicted ):
        try:
            sampe_result = float(
                100 * np.mean(
                    2 * np.abs(predicted - actual) /
                    (np.abs(actual) + np.abs(predicted) + 1e-8)
                )
            )

            return sampe_result
        except Exception as e:
            logger.error(f"SMAPE calculation failed: {str(e)}")
            raise 
    
    def model_training(self):
        try:
            X_train, y_train = self.prepare_data(self.train)
            X_val,   y_val   = self.prepare_data(self.val)

            trained_models = {}

            for model_name, cfg in self.config.models.items():
                model_type   = cfg["type"]
                model_params = cfg["params"]

                if model_type not in MODEL_REGISTRY:
                    logger.warning(
                        f"Skipping {model_name}: '{model_type}' not in registry"
                    )
                    continue

                logger.info("")
                logger.info(f"--- Training {model_name} ({model_type}) ---")
                logger.info("-" * 50)

                model      = get_model(model_type, model_params)
                eval_set   = [(X_val, y_val)]
                fit_kwargs = get_fit_kwargs(model_type, eval_set)

                model.fit(X_train, y_train, **fit_kwargs)


                trained_models[model_name] = {
                    "model"      : model,
                    "model_type" : model_type,
                    "params"     : model_params
                }

                logger.info(f"PASSED - {model_name} trained")

            return trained_models

        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
    
    def save_model(self,model , model_name):

        try:
            
            run_folder = f"model_trained__{self.batch_id}"
            

            local_dir  = os.path.join(self.config.model_dir, run_folder)
            os.makedirs(local_dir, exist_ok=True)
            local_path = os.path.join(local_dir, f"{model_name}.pkl")

            with open(local_path, "wb") as f:
                pickle.dump(model, f)
            logger.info(f"Model saved locally -> {local_path}")

        # ── s3 save ────────────────────────────────────────
            s3_key = f"model_building/{run_folder}/{model_name}.pkl"
            with open(local_path, "rb") as f:
                self.s3.put_object(
                    Bucket      = self.BUCKET_NAME,
                    Key         = s3_key,
                    Body        = f.read(),
                    ContentType = 'application/octet-stream'
                )
            logger.info(f"Model saved to S3  s3://{self.BUCKET_NAME}/{s3_key}")

            logger.info("PASSED - Model saved")
            logger.info("-" * 50)
            return local_path, s3_key

        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise

    # ── log to mlflow ─────────────────────────────────────

    def save_report(self, report):
        try:
            logger.info("")
            logger.info("STEP - SAVING TRAINING REPORT")
            logger.info("-" * 50)

            if not report:
                raise ValueError("Report is empty")

            run_folder = f"model_trained__{self.batch_id}"

            # ── local save ─────────────────────────────────────
            local_dir  = os.path.join(self.config.model_dir, run_folder)
            os.makedirs(local_dir, exist_ok=True)
            local_path = os.path.join(local_dir, "training_report.json")

            with open(local_path, 'w') as f:
                json.dump(report, f, indent=4)
            logger.info(f"Report saved locally -> {local_path}")

            # ── s3 save ────────────────────────────────────────
            s3_key = f"model_building/{run_folder}/training_report.json"
            self.s3.put_object(
                Bucket      = self.BUCKET_NAME,
                Key         = s3_key,
                Body        = json.dumps(report, indent=4),
                ContentType = 'application/json'
            )
            logger.info(f"Report saved to S3 - s3://{self.BUCKET_NAME}/{s3_key}")

            logger.info("PASSED - Training report saved")
            return local_path, s3_key

        except Exception as e:
            logger.error(f"Failed to save training report: {e}")
            raise


    def run(self):
        trained_models = None

        try:
            logger.info("=" * 50)
            logger.info(f'{"=" * 50},MODEL TRAINING PIPELINE STARTED')
            logger.info("=" * 50)

            trained_models = self.model_training()

            logger.info("")
            logger.info("=" * 50)
            logger.info("TRAINING SUMMARY")
            logger.info("=" * 50)
            

            logger.info("")
            logger.info("=" * 50)
            logger.info("SAVING AND LOGGING ALL MODELS")
            logger.info("=" * 50)

            report = {}
            for model_name, result in trained_models.items():
                local_path, s3_key = self.save_model(result["model"], model_name)  # ← unpack two
                report[model_name] = {
                    "model_type" : result["model_type"],
                    "params"     : result["params"],
                    "local_path" : local_path,
                    "s3_key"     : s3_key,
                    "timestamp"  : datetime.now().isoformat()
                }

            self.save_report(report) ## this function creates the model which acts as the input for mext self.log_mlflow
                
            logger.info("")
            logger.info("=" * 50)
            logger.info("MODEL TRAINING PIPELINE COMPLETE")
            logger.info("=" * 50)

            return None

        except Exception as e:
            logger.error(f"Model training pipeline failed: {str(e)}")
            raise

        finally:
            if trained_models is not None:
                for res in trained_models.values():
                    res["model"] = None
                trained_models = None

            self.train = None
            self.val   = None
            gc.collect()
            logger.info("Memory cleared")
            logger.info(f'{"=" * 50},MODEL TRAINING PIPELINE COMPLETED')



             

   



            

   



             

   



            

   