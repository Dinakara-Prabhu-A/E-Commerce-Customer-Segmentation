import os
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

from src.utils import save_object
from src.logger import logging
from src.exception import CustomException
from src.utils import evaluate_models

class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifact", "classifier_model.pkl.gz")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Starting model training process for classification.")

            # Split features and target
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            # Define models and hyperparameter grids
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Decision Tree Classifier": DecisionTreeClassifier(),
                "Random Forest Classifier": RandomForestClassifier(),
                "Gradient Boosting Classifier": GradientBoostingClassifier(),
                "K-Nearest Neighbors Classifier": KNeighborsClassifier(),
               
            }

            param_grids = {
                "Logistic Regression": {"C": [0.1, 1.0, 10.0]},
                "Decision Tree Classifier": {"max_depth": [2, 4, 6], "min_samples_split": [2, 5]},
                "Random Forest Classifier": {"n_estimators": [50, 100], "max_depth": [4, 6, 8]},
                "Gradient Boosting Classifier": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.01, 0.1],
                },
                "K-Nearest Neighbors Classifier": {"n_neighbors": [3, 5, 7]},
                
                }
            

            # Evaluate models and find the best one
            report, best_params = evaluate_models(X_train, y_train, X_test, y_test, models, param_grids, task_type="classification")

            # Identify the best model
            best_model_name = max(report, key=lambda x: report[x]["test_accuracy"])
            best_model = models[best_model_name]
            best_model.set_params(**best_params[best_model_name])
            best_model.fit(X_train, y_train)

            # Save the best model
            save_object(self.model_trainer_config.trained_model_file_path, best_model)
            logging.info(f"Best model saved: {best_model_name}")

            return {
                "best_model_name": best_model_name,
                "best_model_score": report[best_model_name],
                "all_model_reports": report,
            }

        except Exception as e:
            raise CustomException(e, sys)
