import os
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from src.utils import save_object, load_object
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
import nltk

@dataclass
class TextProcessorConfig:
    vectors_file_path: str = os.path.join('artifact', "vectors.pkl")
    df_file_path: str = os.path.join('artifact', "df.pkl")
    vectorizer_file_path: str = os.path.join('artifact', "vectorizer.pkl")

class TextProcessor:
    def __init__(self):
        self.text_processor_config = TextProcessorConfig()

    def read_data(self):
        """
        Reads the description.csv file for the 'Description' column.
        """
        try:
            df = pd.read_csv("artifact/description.csv", usecols=['Description'])
            logging.info("Description data read successfully.")
            return df
        except Exception as e:
            logging.error("Error while reading the data.")
            raise CustomException(e, sys)

    def preprocess_text(self, text):
        """
        Cleans and preprocesses the text by:
        - Removing non-alphabetic characters
        - Converting to lowercase
        - Removing stopwords
        """
        try:
            # Ensure nltk stopwords are downloaded and set up
            nltk.download('stopwords', quiet=True)
            stop_words = set(stopwords.words('english'))

            # Remove non-alphabetic characters and convert to lowercase
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            text = text.lower()
            text = text.strip()

            # Remove stopwords
            text = " ".join([word for word in text.split() if word not in stop_words])
            return text
        except Exception as e:
            logging.error("Error while preprocessing the text.")
            raise CustomException(e, sys)

    def process_and_vectorize(self):
        """
        Preprocesses the data and converts it into vectors using TF-IDF.
        """
        try:
            # Read data
            df = self.read_data()
            
            # Preprocess text
            df['Processed_Description'] = df['Description'].apply(self.preprocess_text)

            # Vectorization using TF-IDF
            vectorizer = TfidfVectorizer()
            vectors = vectorizer.fit_transform(df['Processed_Description'])

            # Save vectors and DataFrame
            save_object(self.text_processor_config.vectors_file_path, vectors)
            df.to_pickle(self.text_processor_config.df_file_path)

            # Save the trained vectorizer
            save_object(self.text_processor_config.vectorizer_file_path, vectorizer)

            logging.info("Vectors, DataFrame, and vectorizer saved successfully.")

            return vectors, df, vectorizer
        except Exception as e:
            logging.error("Error during text processing and vectorization.")
            raise CustomException(e, sys)

    def recommend_for_new_description(self, new_description, vectors, df, vectorizer, top_n=3):
        """
        Recommends top_n similar descriptions to the new description based on cosine similarity.
        """
        try:
            # Preprocess the new description
            processed_description = self.preprocess_text(new_description)

            # Transform the new description using the same vectorizer
            new_description_vector = vectorizer.transform([processed_description])

            # Compute cosine similarity between the new description and existing vectors
            similarity_scores = cosine_similarity(new_description_vector, vectors)

            # Get indices of the top_n most similar descriptions
            similar_indices = similarity_scores.argsort()[0][-top_n:][::-1]

            # Get the top_n recommended descriptions (distinct)
            recommendations = df.iloc[similar_indices]['Description'].unique().tolist()

            logging.info(f"Recommendations for the new description: {recommendations}")
            return recommendations
        except Exception as e:
            logging.error("Error during recommendation generation.")
            raise CustomException(e, sys)

    def load_data(self):
        """
        Loads the previously saved vectors, DataFrame, and vectorizer.
        """
        try:
            vectors = load_object(self.text_processor_config.vectors_file_path)
            df = pd.read_pickle(self.text_processor_config.df_file_path)
            vectorizer = load_object(self.text_processor_config.vectorizer_file_path)
            return vectors, df, vectorizer
        except Exception as e:
            logging.error("Error while loading vectors, DataFrame, and vectorizer.")
            raise CustomException(e, sys)