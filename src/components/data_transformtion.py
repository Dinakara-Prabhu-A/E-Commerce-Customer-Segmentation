import os
import pickle
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler,LabelEncoder,RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin
from src.utils import save_object
from src.logger import logging

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifact', "preprocessor.pkl")

class OutlierIQRWithImputationTransformer(BaseEstimator, TransformerMixin):
    """
    Handles missing values and outliers in specified columns using imputation
    and IQR-based clipping.

    Parameters:
    - columns (list): List of columns to handle.
    - imputer_strategy (str): Strategy for imputation, e.g., 'median'.
    """
    def __init__(self, columns, imputer_strategy='median'):
        self.columns = columns
        self.imputer_strategy = imputer_strategy
        self.imputer = SimpleImputer(strategy=self.imputer_strategy)
        self.lower_thresholds = {}
        self.upper_thresholds = {}

    def fit(self, X, y=None):
        """
        Fits the imputer on the data and calculates the thresholds for outliers based on IQR.
        """
        # Fit the imputer on the selected columns
        self.imputer.fit(X[self.columns])

        # Calculate and save the lower and upper thresholds for IQR-based clipping
        for column in self.columns:
            Q1 = X[column].quantile(0.25)
            Q3 = X[column].quantile(0.75)
            IQR = Q3 - Q1

            lower_threshold = Q1 - 1.5 * IQR
            upper_threshold = Q3 + 1.5 * IQR

            self.lower_thresholds[column] = lower_threshold
            self.upper_thresholds[column] = upper_threshold
        
        return self

    def transform(self, X):
        """
        Imputes missing values and applies outlier clipping.
        """
        df = X.copy()

        # Impute missing values for the specified columns
        df[self.columns] = self.imputer.transform(df[self.columns])

        # Apply the IQR-based clipping for outliers
        for column in self.columns:
            lower_threshold = self.lower_thresholds[column]
            upper_threshold = self.upper_thresholds[column]
            df[column] = df[column].clip(lower=lower_threshold, upper=upper_threshold)

        return df


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):

        try:
            numeric_columns = ['recency','frequency','monetary']

            # Numeric pipeline
            num_pipeline = Pipeline([
                ("outlier_handler", OutlierIQRWithImputationTransformer(columns=numeric_columns, imputer_strategy='median')),
                ('scaler', RobustScaler())
            ])



            # Combine all pipelines into a single ColumnTransformer
            preprocessor = ColumnTransformer([
                ("num", num_pipeline, numeric_columns)
            ])

            return preprocessor
        
        except Exception as e:
            print(f"Error in creating transformer: {str(e)}")
            
    def inititate_data_transformer(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            # Standardize column names
            train_df.columns = train_df.columns.str.strip().str.lower()
            test_df.columns = test_df.columns.str.strip().str.lower()

            # Define the target column name (lowercase after preprocessing)
            target_column_name = 'customersegment'

            # Validate that target column exists
            if target_column_name not in train_df.columns:
                raise Exception(f"Target column '{target_column_name}' not found in training data.")
            if target_column_name not in test_df.columns:
                raise Exception(f"Target column '{target_column_name}' not found in testing data.")
            
            # Split input and target features
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            # Get the preprocessing object
            preprocessing_obj = self.get_data_transformer_object()
            logging.info("Applying preprocessing object on training and testing dataframes.")

            # Apply preprocessing
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine input features and target for train and test arrays
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            # Save the preprocessing object
            save_object(self.data_transformation_config.preprocessor_obj_file_path, preprocessing_obj)
            logging.info("Saved preprocessing object.")
            
            return train_arr, test_arr

        except Exception as e:
            print(f"Error in data transformation: {str(e)}")