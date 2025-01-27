E-commerce Customer Segmentation Project
========================================

This project focuses on customer segmentation and product recommendation in an e-commerce setting using machine learning techniques and NLP.

Overview
--------

The goal of this project is to segment customers based on their purchasing behaviors and recommend products using NLP techniques. It involves preprocessing data, applying unsupervised learning (K-means clustering), and deploying a recommendation system based on cosine similarity of product descriptions.

* * *

Directory Structure
-------------------

```bash
└── e-commerce-customer-segmentation/
    ├── README.md                  # This file
    ├── Dockerfile                 # Docker configuration
    ├── app.py                     # Application entry point
    ├── main.py                    # Main script for execution
    ├── requirements.txt           # Python dependencies
    ├── artifact/                  # Directory for model artifacts and data
    │   ├── classifier_model.pkl.gz   # Example model artifact
    │   ├── data.csv               # Sample dataset
    │   ├── description.csv        # Sample product descriptions
    │   ├── df.pkl                 # Processed data
    │   ├── preprocessor.pkl       # Preprocessing artifact
    │   ├── test.csv               # Test dataset
    │   ├── train.csv              # Training dataset
    │   ├── vectorizer.pkl         # Vectorizer artifact
    │   ├── vectors.pkl            # Vectorized data
    │   └── rfm_data/              # RFM data directory
    │       └── data.csv           # RFM data
    ├── notebook/                  # Jupyter notebooks for EDA and analysis
    │   ├── EDA.ipynb              # Exploratory Data Analysis notebook
    │   └── data.csv               # Sample dataset
    ├── src/                       # Source code directory
    │   ├── __init__.py
    │   ├── components/            # Modules for data processing and modeling
    │   │   ├── __init__.py
    │   │   ├── data_ingestion.py  # Data ingestion module
    │   │   ├── data_transformation.py  # Data transformation module
    │   │   ├── model_trainer.py   # Model training module
    │   │   ├── text_processor.py  # Text processing module
    │   │   └── unsupervised_data.py  # Unsupervised data analysis module
    │   ├── pipeline/              # Pipeline scripts for data flow
    │   │   ├── __init__.py
    │   │   └── predict_pipeline.py  # Prediction pipeline script
    │   ├── exception.py           # Custom exception handling
    │   ├── logger.py              # Logging configuration
    │   └── utils.py               # Utility functions
    └── .streamlit/                # Configuration for Streamlit application
        └── config.toml            # Streamlit configuration
```

* * *

Installation
------------

1.  Clone the repository:
    
    ```bash
    git clone https://github.com/your-username/e-commerce-customer-segmentation.git
    cd e-commerce-customer-segmentation
    ```
    
2.  Install dependencies:
    
    ```bash
    pip install -r requirements.txt
    ```
    

* * *

Usage
-----

To run the application locally:

```bash
python app.py
```

The application will be accessible at `http://localhost:8501` by default.

* * *

Components
----------

*   **Data Ingestion:** `src/components/data_ingestion.py`
*   **Data Transformation:** `src/components/data_transformation.py`
*   **Model Training:** `src/components/model_trainer.py`
*   **Text Processing:** `src/components/text_processor.py`
*   **Unsupervised Data Analysis:** `src/components/unsupervised_data.py`

* * *

Models
------

*   **Classification Models:** Trained classifiers stored as `.pkl` files in `artifact/`

* * *

Text Processing
---------------

*   **NLP Techniques:** Used for text preprocessing and similarity calculation (`text_processor.py`).

* * *

Recommendation System
---------------------

*   **Cosine Similarity:** Implemented to recommend products based on customer-provided descriptions.

* * *

Deployment
----------

*   **Docker:** Containerization setup available (`Dockerfile`).

* * *

Contributing
------------

This project is developed independently. However, feel free to fork the repository, propose changes via pull requests, and report issues.

