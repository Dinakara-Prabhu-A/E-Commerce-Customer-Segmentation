import sys
import pandas as pd
from src.exception import CustomException
from sklearn.exceptions import NotFittedError
from src.utils import load_object
import os

class PredictPipeline:
    def __init__(self):
        self.model_path = 'artifact/classifier_model.pkl.gz'
        self.preprocessor_path = 'artifact/preprocessor.pkl'

    def predict(self, features):
        try:
            # Load the model and preprocessor
            model = load_object(file_path=self.model_path)
            preprocessor = load_object(file_path=self.preprocessor_path)

            # Validate that the model has been fitted
            if not hasattr(model, "predict"):
                raise NotFittedError("The loaded model is not fitted. Train and save the model before using it.")

            # Scale the features and make predictions
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds

        except NotFittedError as e:
            raise CustomException(f"Model Error: {e}", sys)
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(  self,
        recency: int,
        frequency:int,
        monetary:int,
        ):

        self.recency = recency
        self.frequency = frequency
        self.monetary = monetary

        

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "recency": [self.recency],
                "frequency": [self.frequency],
                "monetary": [self.monetary]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)