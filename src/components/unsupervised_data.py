import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from src.logger import logging

class RFMClustering:
    def __init__(self, file_path):
        self.file_path = file_path
        logging.info(f"Initializing RFMClustering with file path: {file_path}")
        self.data = None
        self.rfm_df = None

    def load_data(self):
        try:
            self.data = pd.read_csv(self.file_path, encoding='iso-8859-1')
            logging.info("Data loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise

    def preprocess_data(self):
        try:
            df = self.data.copy()
            df.dropna(subset=['CustomerID'], inplace=True)
            df = df[df['UnitPrice'] != 0]
            df['Quantity'] = df['Quantity'].abs()
            df.drop_duplicates(inplace=True)
            df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
            self.data = df
            logging.info("Data preprocessing completed.")
        except Exception as e:
            logging.error(f"Error during data preprocessing: {e}")
            raise

    def create_rfm_table(self):
        try:
            snapshot_date = self.data['InvoiceDate'].max() + pd.Timedelta(days=1)
            self.data['TotalPrice'] = self.data['Quantity'] * self.data['UnitPrice']
            self.rfm_df = self.data.groupby('CustomerID').agg({
                'InvoiceDate': lambda x: (snapshot_date - x.max()).days,  # Recency
                'InvoiceNo': 'nunique',  # Frequency
                'TotalPrice': 'sum',  # Monetary
            }).reset_index()
            self.rfm_df.drop(columns=['CustomerID'], inplace=True)
            self.rfm_df.rename(columns={
                'InvoiceDate': 'recency',
                'InvoiceNo': 'frequency',
                'TotalPrice': 'monetary'
            }, inplace=True)
            logging.info("RFM table created successfully.")
        except Exception as e:
            logging.error(f"Error creating RFM table: {e}")
            raise

    @staticmethod
    def detect_outliers(column, threshold=1.5):
        q1 = column.quantile(0.25)
        q3 = column.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        return (column < lower_bound) | (column > upper_bound)

    def handle_outliers(self):
        try:
            recency_outliers = self.detect_outliers(self.rfm_df['recency'])
            frequency_outliers = self.detect_outliers(self.rfm_df['frequency'])
            monetary_outliers = self.detect_outliers(self.rfm_df['monetary'])

            non_outliers = ~(recency_outliers | frequency_outliers | monetary_outliers)
            self.non_outlier_rfm_df = self.rfm_df[non_outliers].copy()

            outlier_rfm_recency_df = self.rfm_df[recency_outliers].copy()
            outlier_rfm_frequency_df = self.rfm_df[frequency_outliers].copy()
            outlier_rfm_monetary_df = self.rfm_df[monetary_outliers].copy()

            overlap_indices = outlier_rfm_monetary_df.index.intersection(outlier_rfm_frequency_df.index)

            self.monetary_only_outliers = outlier_rfm_monetary_df.drop(overlap_indices)
            self.frequency_only_outliers = outlier_rfm_frequency_df.drop(overlap_indices)
            self.monetary_and_frequency_outliers = outlier_rfm_monetary_df.loc[overlap_indices]

            logging.info("Outliers handled successfully.")
        except Exception as e:
            logging.error(f"Error handling outliers: {e}")
            raise

    def perform_clustering(self):
        try:
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(self.non_outlier_rfm_df[['monetary', 'frequency', 'recency']])

            svd = TruncatedSVD(n_components=2)
            reduced_features = svd.fit_transform(scaled_data)

            kmeans = KMeans(n_clusters=3, random_state=42, max_iter=1000)
            cluster_labels = kmeans.fit_predict(reduced_features)
            self.non_outlier_rfm_df['Cluster'] = cluster_labels

            self.monetary_only_outliers['Cluster'] = 4
            self.frequency_only_outliers['Cluster'] = 5
            self.monetary_and_frequency_outliers['Cluster'] = 6

            outlier_clusters_df = pd.concat([
                self.monetary_only_outliers,
                self.frequency_only_outliers,
                self.monetary_and_frequency_outliers
            ])

            self.full_clustering_df = pd.concat([
                self.non_outlier_rfm_df, outlier_clusters_df
            ])

            # cluster_mapping = {
            #     0: "Potential",
            #     1: "Frequent",
            #     2: "Loyal",
            #     4: "Inconsistent",
            #     5: "Dormant",
            #     6: "Bulk Purchase"
            # }
            self.full_clustering_df.rename(columns={'Cluster': 'customersegment'}, inplace=True)
            # self.full_clustering_df['CustomerSegment'] = self.full_clustering_df['CustomerSegment'].map(cluster_mapping)

            logging.info("Clustering performed successfully.")
        except Exception as e:
            logging.error(f"Error performing clustering: {e}")
            raise

    def save_results(self, output_path):
        try:
            self.full_clustering_df.to_csv(output_path, index=False)
            logging.info(f"Results saved to {output_path}.")
        except Exception as e:
            logging.error(f"Error saving results: {e}")
            raise
        
    def save_description_column(self, output_path):
        try:
            description_column = self.data[['Description']].dropna()
            description_column.to_csv(output_path, index=False, header=True)
            logging.info(f"'Description' column saved to {output_path}.")
        except Exception as e:
            logging.error(f"Error saving 'Description' column: {e}")
            raise

    def execute(self, output_path):
        try:
            self.load_data()
            self.preprocess_data()
            self.save_description_column('artifact/description.csv')  # Save 'Description' column
            self.create_rfm_table()
            self.handle_outliers()
            self.perform_clustering()
            self.save_results(output_path)
            logging.info("RFM clustering process completed successfully.")
        except Exception as e:
            logging.error(f"Error executing RFM clustering: {e}")
            raise

