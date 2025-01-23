from src.utils import UnSupervisedData
from src.components.data_ingestion import DataIngestion
from src.components.data_transformtion import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.text_processor import TextProcessor, TextProcessorConfig
from nltk import download

if __name__ == "__main__":
    
    # unsupervised_data = UnSupervisedData()
    # unsupervised_data.execute()
    # data_ingestion = DataIngestion()
    # train_data,test_data = data_ingestion.initiate_data_ingestion()
    # data_transformation = DataTransformation()
    # train_arr, test_arr = data_transformation.inititate_data_transformer(train_data, test_data)
    # model_trainer = ModelTrainer()
    # model_trainer.initiate_model_trainer(train_arr, test_arr)
    download("stopwords")

    # Define the configuration
    config = TextProcessorConfig(csv_file_path="artifact/description.csv")

    # Initialize the TextProcessor
    processor = TextProcessor(config)

    # # Step 1: Fit and save the vectorizer
    # processor.fit_and_save_vectorizer()

    # Step 2: Recommend top N similar descriptions for a sample text
    sample_text = "This is an amazing product with excellent features!"
    recommendations = processor.recommend_top_n(sample_text, n=3)
    print("Top Recommendations:")
    for idx, rec in enumerate(recommendations, 1):
        print(f"{rec}")
