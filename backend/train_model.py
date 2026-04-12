import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
import os

def train_segmentation_model():
    print("Loading data...")
    df = pd.read_csv('data/online_retail.csv', parse_dates=['InvoiceDate'])
    
    # Calculate Total Price
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    
    # RFM Analysis
    print("Performing RFM Analysis...")
    snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'InvoiceNo': 'count',
        'TotalPrice': 'sum'
    })
    
    rfm.rename(columns={
        'InvoiceDate': 'Recency',
        'InvoiceNo': 'Frequency',
        'TotalPrice': 'Monetary'
    }, inplace=True)
    
    # Prune outliers (optional but good for K-Means)
    # For synthetic data, it might not be necessary, but let's keep it robust
    
    # Preprocessing
    print("Scaling features...")
    features = ['Recency', 'Frequency', 'Monetary']
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[features])
    
    # Training K-Means
    print("Training K-Means (K=4)...")
    kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
    
    # Save model and scaler
    os.makedirs('backend', exist_ok=True)
    joblib.dump(kmeans, 'backend/model.pkl')
    joblib.dump(scaler, 'backend/scaler.pkl')
    
    # Save the RFM results for reference
    rfm.to_csv('data/rfm_segmented.csv')
    print("Model and scaler saved in backend/ folder.")
    print("RFM results saved in data/rfm_segmented.csv")

if __name__ == "__main__":
    train_segmentation_model()
