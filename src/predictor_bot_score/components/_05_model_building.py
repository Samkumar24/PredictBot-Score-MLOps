from dataclasses import dataclass
from pathlib import Path
from src.predictor_bot_score.config.configuration import yaml_load , create_directories
from src.predictor_bot_score.logger import logger
from src.predictor_bot_score.constants import CONFIG_PATH
from src.predictor_bot_score.entity import ModelTrainingConfig
import os
import pandas as pd
import glob
import mlflow
import lightgbm
import pickle
from datetime import datetime
import sqlite3

class Model_Building :

    def __init__(self, config : ModelTrainingConfig):
        self.config = config
        self.train , self.val = self._read_data()

    def _get_files(self , folder:Path ,prefix :str):
        files = glob.glob(os.path.join(folder ,f"{prefix}_*.csv"))
        if not files:
            raise FileNotFoundError(f"No {prefix} file found in {folder}")
        return max(files , key=os.path.getmtime)
    

    def _read_data(self):
        try:
            logger.info("=" * 50)
            logger.info("Reading train /  test splits")
            logger.info("=" * 50)

            train_file = self._get_files(self.config.train_data_path, "train")
            val_file   = self._get_files(self.config.val_data_path,   "val")

            train = pd.read_csv(train_file)
            val   = pd.read_csv(val_file)
            
            logger.info(f"Train : {len(train)} rows")
            logger.info(f"Val   : {len(val)} rows")

            return train, val

        except FileNotFoundError as e:
            logger.error(f"Split file not found: {e}")
            raise

        except Exception as e:
            logger.error(f"Failed to read splits: {str(e)}")
            raise

    def prepare_Data(self ,df :pd.DataFrame):
        try:
            X = df[self.config.features]
            y = df[self.config.target_column]

            return X ,y
        except Exception as e:
            raise e 
    
    def model_training(self):

        try:

            X_train, y_train = self.prepare_Data(self.train)
            X_val,   y_val   = self.prepare_Data(self.val)

            model = lightgbm.LGBMRegressor(**self.config.lgbm_params)
            model.fit(
                X_train , y_train,
                eval_set=[(X_val , y_val)],
                callbacks = [
                    lightgbm.early_stopping(100, verbose=False)
                   
                ]
            )

            logger.info(f"Best iteration : {model.best_iteration_}")
            logger.info("PASSED - Model trained")
            logger.info("-" * 50)

            return model

        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
    
    def save_model(self,model):

        try:
            path = self.config.model_dir
            time_stamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
            model_path = os.path.join(path,f"{time_stamp}.pkl")

            with open(model_path ,"wb") as f:
                pickle.dump(model , f)
        
            logger.info(f"Model saved ")
            logger.info("PASSED - Model saved")
            logger.info("-" * 50)

            return model_path

        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            raise

    # ── log to mlflow ─────────────────────────────────────
    def log_to_mlflow(self, model, model_path):
        try:
            logger.info("")
            logger.info("STEP 3 - LOGGING TO MLFLOW")
            logger.info("-" * 50)

            mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
            mlflow.set_experiment(self.config.mlflow_experiment)

            with mlflow.start_run():

                mlflow.log_params(self.config.lgbm_params)

                mlflow.set_tag("stage",      "Staging")
                mlflow.set_tag("trained_at", datetime.now().isoformat())
                mlflow.set_tag("model_path", model_path)


                mlflow.lightgbm.log_model(
                    model,
                    "model",
                    registered_model_name="Predicted_model_name"

                )
            logger.info("PASSED - MLflow logging complete")
            logger.info("-" * 50)

        except Exception as e:
            logger.error(f"MLflow logging failed: {str(e)}")
            raise

    def run(self):
        
        try:
            logger.info("=" * 50)
            logger.info("MODEL TRAINING PIPELINE STARTED")
            logger.info("=" * 50)

            model = self.model_training()
            model_save = self.save_model(model)
            self.log_to_mlflow(model , model_path=model_save)

            logger.info("=" * 50)
            logger.info("MODEL TRAINING PIPELINE COMPLETE")
            logger.info("=" * 50)

        except Exception as e:
            logger.error(f"Model training pipeline failed: {str(e)}")
            raise



            

   