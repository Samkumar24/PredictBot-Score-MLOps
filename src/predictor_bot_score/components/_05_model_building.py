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
from datetime import datetime
import sqlite3
import gc
from src.predictor_bot_score.entity import ModelTrainingConfig



class Model_Building :

    def __init__(self, config : ModelTrainingConfig):
        self.config = config
        self.train , self.val = self._read_data()
        self.batch_id = datetime.now().strftime("%Y_%m_%d_%H") 

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
            path = self.config.model_dir
            time_stamp = datetime.now().strftime("%Y_%m_%d_%H")
            

            run_dir = os.path.join(path ,f'run__{time_stamp}')
            os.makedirs(run_dir , exist_ok=True)

            model_path = os.path.join(run_dir,f"{model_name}.pkl")
            
            with open(model_path ,"wb") as f:
                pickle.dump(model , f)
                f.close()
        
            logger.info(f"Model saved  {model_path}")
            logger.info("PASSED - Model saved")
            logger.info("-" * 50)

            return model_path

        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            raise

    # ── log to mlflow ─────────────────────────────────────


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

            for model_name, result in trained_models.items():
                
                model_path = self.save_model(result["model"], model_name) ## this function creates the model which acts as the input for mext self.log_mlflow
                
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



             

   



            

   



             

   



            

   