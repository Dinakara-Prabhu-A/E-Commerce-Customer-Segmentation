import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords

from dataclasses import dataclass
from src.utils import save_object, load_object

@dataclass
class TextProcessorConfig:
    csv_file_path: str  # Path to the CSV file containing descriptions
    pickle_file_path: str = "artifact/tfidf_vectorizer.pkl.gz"  # Path to save the TF-IDF vectorizer
    column_name: str = "Description"  # Column containing the descriptions


class TextProcessor:
    def __init__(self, config: TextProcessorConfig):
        self.config = config
        self.vectorizer = None
        self.tfidf_matrix = None

    def custom_preprocessor(self, text: str) -> str:
        """
        Preprocess text: make lowercase, remove non-alphabetic characters, and remove stopwords.

        Args:
            text (str): Input text.

        Returns:
            str: Cleaned text.
        """
        # Convert text to lowercase
        text = text.lower()
        # Remove non-alphabetic characters
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        # Remove stopwords
        stop_words = set(stopwords.words("english"))
        text = " ".join(word for word in text.split() if word not in stop_words)
        return text

    def fit_and_save_vectorizer(self):
        """
        Fit the TF-IDF vectorizer on the descriptions column and save the model.
        """
        # Load the CSV file
        data = pd.read_csv(self.config.csv_file_path)
        if self.config.column_name not in data.columns:
            raise ValueError(f"Column '{self.config.column_name}' not found in the dataset.")

        # Drop missing values
        data = data.dropna(subset=[self.config.column_name])

        # Preprocess the descriptions
        descriptions = data[self.config.column_name].apply(self.custom_preprocessor).tolist()

        # Initialize and fit the TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(descriptions)

        # Save the vectorizer
        save_object(self.config.pickle_file_path, self.vectorizer)
        print(f"TF-IDF vectorizer saved at {self.config.pickle_file_path}")

    def recommend_top_n(self, input_text: str, n: int = 3):
        """
        Recommend top N distinct descriptions based on cosine similarity.

        Args:
            input_text (str): Input text to compare.
            n (int): Number of recommendations.

        Returns:
            List[str]: Top N recommended descriptions.
        """
        # Load the vectorizer
        if not self.vectorizer:
            self.vectorizer = load_object(self.config.pickle_file_path)

        # Preprocess the input text
        cleaned_text = self.custom_preprocessor(input_text)

        # Transform the input text
        input_vector = self.vectorizer.transform([cleaned_text])

        # Compute cosine similarity
        cosine_similarities = cosine_similarity(input_vector, self.tfidf_matrix).flatten()

        # Get indices of top N distinct recommendations
        top_indices = cosine_similarities.argsort()[-n:][::-1]

        # Return the top N recommendations
        data = pd.read_csv(self.config.csv_file_path).dropna(subset=[self.config.column_name])
        return data.iloc[top_indices][self.config.column_name].tolist()


    