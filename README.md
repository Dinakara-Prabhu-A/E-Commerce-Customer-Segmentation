# E-Commerce Customer Segmentation Project

This project focuses on customer segmentation and product recommendation using machine learning techniques, specifically clustering and classification.

## Project Overview

The goal of this project is to segment customers based on their purchasing behavior and demographics using unsupervised learning techniques. Subsequently, the segmented clusters are used as target labels for supervised classification, predicting customer segments for new data.

## Directory Structure

```
└── dinakara-prabhu-a-e-commerce-customer-segmentation/
    ├── README.md               # Project overview and setup instructions
    ├── Dockerfile              # Docker configuration for containerization
    ├── app.py                  # Streamlit web application for model deployment
    ├── main.py                 # Main script for executing the project
    ├── requirements.txt        # Python dependencies
    ├── artifact/               # Directory for storing project artifacts
    │   ├── classifier_model.pkl.gz   # Saved Random Forest Classifier model
    │   ├── data.csv            # Raw data containing customer information
    │   ├── description.csv     # Raw product descriptions
    │   ├── df.pkl              # Processed data after feature engineering
    │   ├── preprocessor.pkl    # Serialized preprocessing steps
    │   ├── test.csv            # Test dataset
    │   ├── train.csv           # Training dataset
    │   ├── vectorizer.pkl      # Saved vectorizer for product descriptions
    │   ├── vectors.pkl         # Vectors for product descriptions
    │   └── rfm_data/           # Directory for RFM data storage
    │       └── data.csv        # RFM transformed data
    ├── notebook/               # Jupyter notebooks for exploratory data analysis (EDA)
    │   ├── EDA.ipynb           # Notebook documenting data exploration
    │   └── data.csv            # Dataset used in the notebooks
    ├── src/                    # Source code directory
    │   ├── __init__.py         # Initialization file
    │   ├── exception.py        # Custom exception handling
    │   ├── logger.py           # Logging configuration
    │   ├── utils.py            # Utility functions
    │   ├── components/         # Directory for project components
    │   │   ├── __init__.py     # Initialization file for components
    │   │   ├── data_ingestion.py  # Script for data ingestion
    │   │   ├── data_transformation.py  # Script for data preprocessing
    │   │   ├── model_trainer.py  # Script for training machine learning models
    │   │   ├── text_processor.py  # Script for NLP text processing
    │   │   └── unsupervised_data.py  # Script for unsupervised learning
    │   └── pipeline/           # Directory for machine learning pipeline
    │       ├── __init__.py     # Initialization file for pipeline
    │       └── predict_pipeline.py  # Pipeline for prediction
    └── .streamlit/             # Streamlit configuration directory
        └── config.toml         # Streamlit configuration file
```

## Project Details

### Workflow

1. **Data Ingestion and Preprocessing**: Raw data is ingested and preprocessed. This includes handling missing values, scaling numerical features, and encoding categorical variables.

2. **Unsupervised Learning**: Using unsupervised learning techniques (clustering), customers are segmented into distinct groups based on their behavior.

3. **Feature Engineering**: Features are engineered from the raw data, including RFM (Recency, Frequency, Monetary) features, which are crucial for customer segmentation.

4. **Supervised Learning**: The segmented clusters are used as labels for supervised learning. Various classification algorithms are trained and evaluated to predict customer segments for new data.

5. **NLP for Product Recommendations**: Product descriptions are processed using NLP techniques to vectorize text data. Cosine similarity is used to recommend products based on customer segments.

6. **Model Deployment**: A Streamlit web application (`app.py`) is provided for deploying the model and showcasing the product recommendation system.

### Model Selection

The Random Forest Classifier was selected as the best-performing model for customer segmentation and classification.

### Usage

To run the project locally, ensure you have Python installed along with the dependencies listed in `requirements.txt`. You can then execute `main.py` to reproduce the data pipeline and model training. For model deployment, use `streamlit run app.py`.

## Getting Started

1. Clone the repository:

   ```bash
   git clone <repository_url>
   cd dinakara-prabhu-a-e-commerce-customer-segmentation
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the project:

   ```bash
   python main.py
   ```

4. Access the Streamlit web app locally:

   ```bash
   streamlit run app.py
   ```

## Author

Dinakara Prabhu

