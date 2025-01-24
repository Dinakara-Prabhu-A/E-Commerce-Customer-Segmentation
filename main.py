from src.utils import UnSupervisedData
from src.components.data_ingestion import DataIngestion
from src.components.data_transformtion import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.text_processor import TextProcessor

if __name__ == "__main__":
    
    unsupervised_data = UnSupervisedData()
    unsupervised_data.execute()
    data_ingestion = DataIngestion()
    train_data,test_data = data_ingestion.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr, test_arr = data_transformation.inititate_data_transformer(train_data, test_data)
    model_trainer = ModelTrainer()
    model_trainer.initiate_model_trainer(train_arr, test_arr)
    processor = TextProcessor()

    # Step 1: Process and vectorize the data
    vectors, df, vectorizer = processor.process_and_vectorize()

    print("Data processed and saved successfully.")