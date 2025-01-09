import os
import sys
import pickle
import pandas as pd
import time
from sklearn.model_selection import GridSearchCV
from src.components.unsupervised_data import RFMClustering
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    r2_score,
    mean_absolute_error,
    mean_squared_error
)
from dataclasses import dataclass
import warnings
from src.exception import CustomException
from src.logger import logging
import gzip

warnings.filterwarnings("ignore")  # Ignore warnings for cleaner output


def save_object(file_path, obj):
    """Save an object to a compressed pickle file using gzip."""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with gzip.open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """Load an object from a compressed pickle file using gzip."""
    try:
        with gzip.open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, param_grids, task_type="classification"):
    """
    Evaluate multiple models using GridSearchCV, calculate metrics, and log all results.

    Parameters:
    - X_train, y_train: Training dataset features and target.
    - X_test, y_test: Testing dataset features and target.
    - models: Dictionary of models to evaluate.
    - param_grids: Dictionary of hyperparameter grids for each model.
    - task_type: 'classification' or 'regression'.

    Returns:
    - report: Dictionary with all metrics for each model.
    - best_params: Dictionary of the best parameters for each model.
    """
    try:
        report = {}
        best_params = {}

        for model_name, model in models.items():
            logging.info(f"Training and evaluating model: {model_name}")
            
            # Get hyperparameters for the model
            params = param_grids.get(model_name, {})
            
            # Perform GridSearchCV for hyperparameter tuning
            gs = GridSearchCV(model, params, cv=3, scoring="accuracy" if task_type == "classification" else "r2", n_jobs=-1)
            gs.fit(X_train, y_train)

            # Log and store the best parameters
            best_params[model_name] = gs.best_params_
            logging.info(f"Best parameters for {model_name}: {best_params[model_name]}")

            # Set the best parameters and fit the model
            model.set_params(**best_params[model_name])
            model.fit(X_train, y_train)

            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            if task_type == "classification":
                # Classification Metrics
                metrics = {
                    "train_accuracy": accuracy_score(y_train, y_train_pred),
                    "test_accuracy": accuracy_score(y_test, y_test_pred),
                }

            elif task_type == "regression":
                # Regression Metrics
                metrics = {
                    "train_r2": r2_score(y_train, y_train_pred),
                    "test_r2": r2_score(y_test, y_test_pred),
                }

            # Log the metrics
            logging.info(f"Metrics for {model_name}: {metrics}")
            logging.info("-" * 50)

            # Store the metrics for the model
            report[model_name] = metrics

        return report, best_params

    except Exception as e:
        raise CustomException(e, sys)


class UnSupervisedData:
    def __init__(self):
        self.input_file_path = 'notebook/data.csv'  # Path to the input CSV file
        self.output_file_path = 'artifact/rfm_data/data.csv'  # Path to save the results
    
    def execute(self):
        try:
            rfm_clustering = RFMClustering(file_path=self.input_file_path)
            rfm_clustering.execute(output_path=self.output_file_path)
            print(f"RFM clustering results saved to: {self.output_file_path}")
        except Exception as e:
            print(f"An error occurred: {e}")
        
    

class RawData:

    def __init__(self,file_path):
        self.data = pd.read_excel(file_path)

    def regression_data(self):
        df = self.data.copy()
        df.drop(df[df['selling_price'] < 1].index, inplace=True)
        df.dropna(subset=['selling_price'], inplace=True)
        df['item_date'] = pd.to_datetime(df['item_date'],format='%Y%m%d',errors='coerce')
        df['delivery date'] = pd.to_datetime(df['delivery date'],format = '%Y%m%d',errors='coerce')
        df['item_day'] = df['item_date'].dt.day
        df['item_month'] = df['item_date'].dt.month
        df['item_year'] = df['item_date'].dt.year
        df['delivery_day'] = df['delivery date'].dt.day
        df['delivery_month'] = df['delivery date'].dt.month
        df['delivery_year'] = df['delivery date'].dt.year
        df['quantity tons'] = pd.to_numeric(df['quantity tons'],errors='coerce')
        df.drop(columns = ['material_ref','id','customer','delivery date','item_date'],inplace = True)
        
        return df
    
    def classification_data(self):
        df = self.data.copy()
        df = df[df['status'].isin(['Won', 'Lost'])]
        df['status'] = df['status'].replace({'Won': 0, 'Lost': 1})
        df['item_date'] = pd.to_datetime(df['item_date'],format='%Y%m%d',errors='coerce')
        df['delivery date'] = pd.to_datetime(df['delivery date'],format = '%Y%m%d',errors='coerce')
        df['item_day'] = df['item_date'].dt.day
        df['item_month'] = df['item_date'].dt.month
        df['item_year'] = df['item_date'].dt.year
        df['delivery_day'] = df['delivery date'].dt.day
        df['delivery_month'] = df['delivery date'].dt.month
        df['delivery_year'] = df['delivery date'].dt.year
        df['quantity tons'] = pd.to_numeric(df['quantity tons'],errors='coerce')
        df.drop(columns = ['material_ref','id','customer','delivery date','item_date'],inplace = True)
        
        return df