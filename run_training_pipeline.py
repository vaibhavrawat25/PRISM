import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from lifetimes.utils import summary_data_from_transaction_data
from lifetimes import BetaGeoFitter, GammaGammaFitter
import joblib
import os
import json
from datetime import datetime

# --- Configuration ---
# Input Data
RAW_DATA_PATH = 'data/online_retail.csv'

# Output Artifacts
SEGMENTED_DATA_PATH = 'data/rfm_segmented.csv'
MODEL_DIR = 'backend'
LOG_FILE = 'training_log.json'

# --- Helper Functions ---
def print_header(title):
    """Prints a formatted header to the console."""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def log_results(log_data):
    """Saves the training log to a JSON file."""
    with open(LOG_FILE, 'w') as f:
        json.dump(log_data, f, indent=4)
    print(f"\n[SUCCESS] Training log saved to {LOG_FILE}")

# --- Pipeline Steps ---

def step_1_run_segmentation(df):
    """
    Performs RFM analysis and K-Means clustering to segment customers.
    Saves the segmentation model, scaler, persona map, and the segmented data.
    """
    print_header("Step 1: Running Customer Segmentation (K-Means)")

    # a. RFM Aggregation
    print("  - Calculating RFM values...")
    snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalPrice': 'sum'
    }).rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'TotalPrice': 'Monetary'})

    # b. Preprocessing
    print("  - Applying log transformation and scaling...")
    rfm_log = np.log1p(rfm[['Recency', 'Frequency', 'Monetary']])
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_log)

    # c. K-Means Clustering
    print("  - Training K-Means model (k=4)...")
    kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42, n_init=10)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

    # d. Persona Mapping
    print("  - Generating persona map...")
    cluster_stats = rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean().sort_values(by='Monetary', ascending=False)
    persona_names = ["Champions", "Potential Loyalists", "At-Risk Customers", "Lost Customers"]
    persona_map = {cluster_id: persona_names[i] for i, cluster_id in enumerate(cluster_stats.index)}

    # e. Save Artifacts
    print("  - Saving segmentation artifacts...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(kmeans, os.path.join(MODEL_DIR, 'model.pkl'))
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
    joblib.dump(persona_map, os.path.join(MODEL_DIR, 'persona_map.pkl'))
    rfm.to_csv(SEGMENTED_DATA_PATH)
    
    print("[SUCCESS] Step 1 complete.")
    return rfm

def step_2_run_churn_modeling(rfm_df):
    """
    Trains a churn prediction model using the segmented RFM data.
    Saves the churn model and its scaler.
    """
    print_header("Step 2: Running Churn Prediction (Logistic Regression)")

    # a. Define Churn
    recency_threshold = rfm_df['Recency'].quantile(0.75)
    rfm_df['Churn'] = np.where(rfm_df['Recency'] > recency_threshold, 1, 0)
    print(f"  - Churn defined as Recency > {recency_threshold:.0f} days.")

    # b. Feature Selection and Splitting
    features = ['Recency', 'Frequency', 'Monetary']
    X = rfm_df[features]
    y = rfm_df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # c. Scaling
    print("  - Scaling features...")
    churn_scaler = StandardScaler()
    X_train_scaled = churn_scaler.fit_transform(X_train)
    X_test_scaled = churn_scaler.transform(X_test)

    # d. Model Training
    print("  - Training Logistic Regression model...")
    churn_model = LogisticRegression(random_state=42, class_weight='balanced')
    churn_model.fit(X_train_scaled, y_train)

    # e. Evaluation
    y_pred = churn_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"  - Model Evaluation Accuracy: {accuracy:.2f}")

    # f. Save Artifacts
    print("  - Saving churn model artifacts...")
    joblib.dump(churn_model, os.path.join(MODEL_DIR, 'churn_model.pkl'))
    joblib.dump(churn_scaler, os.path.join(MODEL_DIR, 'churn_scaler.pkl'))
    
    print("[SUCCESS] Step 2 complete.")
    return accuracy

def step_3_run_ltv_modeling(df):
    """
    Trains BG/NBD and Gamma-Gamma models for LTV prediction.
    Saves the trained models.
    """
    print_header("Step 3: Running LTV Forecasting (BG/NBD & Gamma-Gamma)")

    # a. Data Preparation for Lifetimes library
    print("  - Formatting data for LTV models...")
    summary = summary_data_from_transaction_data(
        df,
        customer_id_col='CustomerID',
        datetime_col='InvoiceDate',
        monetary_value_col='TotalPrice',
        observation_period_end=df['InvoiceDate'].max()
    )

    # b. Train BG/NBD Model
    print("  - Training BG/NBD model to predict future transactions...")
    bgnbd = BetaGeoFitter(penalizer_coef=0.001)
    bgnbd.fit(summary['frequency'], summary['recency'], summary['T'])

    # c. Train Gamma-Gamma Model
    print("  - Training Gamma-Gamma model to predict transaction value...")
    returning_customers_summary = summary[summary['frequency'] > 0]
    ggf = GammaGammaFitter(penalizer_coef=0.001)
    ggf.fit(returning_customers_summary['frequency'], returning_customers_summary['monetary_value'])

    # d. Save Artifacts
    print("  - Saving LTV model artifacts...")
    bgnbd.save_model(os.path.join(MODEL_DIR, 'bgnbd_model.pkl'))
    ggf.save_model(os.path.join(MODEL_DIR, 'ggf_model.pkl'))
    
    print("[SUCCESS] Step 3 complete.")

# --- Main Orchestrator ---
def main():
    """
    Main function to run the entire training pipeline.
    """
    print_header("PRISM AI/ML Training Pipeline Started")
    
    # --- Load Data ---
    print(f"  - Loading raw data from {RAW_DATA_PATH}...")
    try:
        # Read and preprocess raw data once
        df = pd.read_csv(RAW_DATA_PATH, encoding='ISO-8859-1')
        df = df.dropna(subset=['CustomerID'])
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
        df = df.dropna(subset=['InvoiceDate'])
        df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    except FileNotFoundError:
        print(f"[ERROR] Raw data file not found at {RAW_DATA_PATH}. Aborting.")
        return

    # --- Run Pipeline Steps ---
    rfm_data = step_1_run_segmentation(df.copy())
    churn_accuracy = step_2_run_churn_modeling(rfm_data)
    step_3_run_ltv_modeling(df.copy())

    # --- Log Results ---
    training_log = {
        "last_trained_timestamp": datetime.now().isoformat(),
        "models_trained": ["KMeans", "LogisticRegression_Churn", "BG/NBD_LTV", "GammaGamma_LTV"],
        "churn_model_accuracy": round(churn_accuracy, 4)
    }
    log_results(training_log)
    
    print_header("PRISM AI/ML Training Pipeline Finished Successfully")

if __name__ == "__main__":
    main()
