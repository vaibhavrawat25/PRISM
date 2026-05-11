from flask import Flask, render_template, jsonify, request, send_file
import pandas as pd
import joblib
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data

app = Flask(__name__, template_folder='../templates', static_folder='../static')
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Paths
MODEL_PATH = 'backend/model.pkl'
SCALER_PATH = 'backend/scaler.pkl'
MAP_PATH = 'backend/persona_map.pkl'
CHURN_MODEL_PATH = 'backend/churn_model.pkl'
CHURN_SCALER_PATH = 'backend/churn_scaler.pkl'
BGNBD_MODEL_PATH = 'backend/bgnbd_model.pkl'
GGF_MODEL_PATH = 'backend/ggf_model.pkl'
DATA_PATH = 'data/rfm_segmented.csv'
TRANSACTION_COLUMNS = {'InvoiceNo', 'InvoiceDate', 'Quantity', 'UnitPrice', 'CustomerID'}
RFM_COLUMNS = {'CustomerID', 'Recency', 'Frequency', 'Monetary'}

# Persistent objects
model = None
scaler = None
persona_map = None
churn_model = None
churn_scaler = None
bgnbd_model = None
ggf_model = None

def load_persistence():
    global model, scaler, persona_map, churn_model, churn_scaler, bgnbd_model, ggf_model
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
    if os.path.exists(MAP_PATH):
        persona_map = joblib.load(MAP_PATH)
    if os.path.exists(CHURN_MODEL_PATH) and os.path.exists(CHURN_SCALER_PATH):
        churn_model = joblib.load(CHURN_MODEL_PATH)
        churn_scaler = joblib.load(CHURN_SCALER_PATH)
    if os.path.exists(BGNBD_MODEL_PATH) and os.path.exists(GGF_MODEL_PATH):
        bgnbd_model = BetaGeoFitter()
        bgnbd_model.load_model(BGNBD_MODEL_PATH)
        ggf_model = GammaGammaFitter()
        ggf_model.load_model(GGF_MODEL_PATH)

load_persistence()

def _normalize_columns(df):
    df = df.copy()
    df.columns = df.columns.str.strip()
    return df


def _cluster_rfm(rfm):
    if rfm.empty:
        raise ValueError("No valid customer rows found after cleaning the CSV.")

    rfm = rfm.copy()
    rfm = rfm.replace([np.inf, -np.inf], np.nan).dropna(subset=['Recency', 'Frequency', 'Monetary'])
    rfm = rfm[(rfm['Recency'] >= 0) & (rfm['Frequency'] > 0) & (rfm['Monetary'] > 0)]
    if rfm.empty:
        raise ValueError("CSV must contain customers with positive frequency and monetary values.")

    rfm_log = rfm[['Recency', 'Frequency', 'Monetary']].copy()
    for col in rfm_log.columns:
        rfm_log[col] = np.log1p(rfm_log[col])

    scaler_new = StandardScaler()
    scaled_data = scaler_new.fit_transform(rfm_log)

    k = min(4, len(rfm))
    kmeans_new = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    clusters = kmeans_new.fit_predict(scaled_data)
    rfm['Cluster'] = clusters.astype(int)

    cluster_stats = rfm.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean'
    })
    cluster_stats['score'] = (
        cluster_stats['Frequency'].rank(method='first') +
        cluster_stats['Monetary'].rank(method='first') -
        cluster_stats['Recency'].rank(method='first')
    )
    cluster_stats = cluster_stats.sort_values(by='score', ascending=False)

    persona_names = ["Champions", "Potential Loyalists", "At-Risk Customers", "Lost Customers"]
    new_map = {int(cluster_id): persona_names[i] for i, cluster_id in enumerate(cluster_stats.index)}

    joblib.dump(kmeans_new, MODEL_PATH)
    joblib.dump(scaler_new, SCALER_PATH)
    joblib.dump(new_map, MAP_PATH)
    rfm.to_csv(DATA_PATH, index=False)
    load_persistence()
    return rfm


def process_transactions(df):
    df = _normalize_columns(df)
    missing = sorted(TRANSACTION_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"Missing required transaction columns: {', '.join(missing)}")

    df = df.dropna(subset=['CustomerID']).copy()
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
    df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
    df['UnitPrice'] = pd.to_numeric(df['UnitPrice'], errors='coerce')
    df['CustomerID'] = pd.to_numeric(df['CustomerID'], errors='coerce')
    df = df.dropna(subset=['InvoiceDate', 'Quantity', 'UnitPrice', 'CustomerID'])
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    df['CustomerID'] = df['CustomerID'].astype(int)
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

    snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalPrice': 'sum'
    }).reset_index()
    rfm.rename(columns={
        'InvoiceDate': 'Recency',
        'InvoiceNo': 'Frequency',
        'TotalPrice': 'Monetary'
    }, inplace=True)
    return _cluster_rfm(rfm)


def process_precomputed_rfm(df):
    df = _normalize_columns(df)
    missing = sorted(RFM_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"Missing required RFM columns: {', '.join(missing)}")

    rfm = df[['CustomerID', 'Recency', 'Frequency', 'Monetary']].copy()
    for col in ['CustomerID', 'Recency', 'Frequency', 'Monetary']:
        rfm[col] = pd.to_numeric(rfm[col], errors='coerce')
    rfm = rfm.dropna(subset=['CustomerID', 'Recency', 'Frequency', 'Monetary'])
    rfm['CustomerID'] = rfm['CustomerID'].astype(int)
    rfm = rfm.groupby('CustomerID', as_index=False).agg({
        'Recency': 'min',
        'Frequency': 'sum',
        'Monetary': 'sum'
    })
    return _cluster_rfm(rfm)


def process_rfm(df):
    """Accept raw transaction data or a precomputed CustomerID/RFM CSV."""
    normalized = _normalize_columns(df)
    if RFM_COLUMNS.issubset(normalized.columns):
        return process_precomputed_rfm(normalized)
    return process_transactions(normalized)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/sample_csv')
def sample_csv():
    return send_file(
        os.path.join(BASE_DIR, 'data', 'user_test.csv'),
        mimetype='text/csv',
        as_attachment=True,
        download_name='prism_sample_transactions.csv'
    )

@app.route('/upload', methods=['POST'])
def upload_data():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and file.filename.lower().endswith('.csv'):
        try:
            df = pd.read_csv(file, encoding='ISO-8859-1')
            rfm = process_rfm(df)
            return jsonify({
                "message": "File processed successfully",
                "customers": int(rfm['CustomerID'].nunique())
            })
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify({"error": "Invalid file type"}), 400

@app.route('/segment')
def get_segments():
    if not os.path.exists(DATA_PATH) or persona_map is None:
        return jsonify({"error": "empty_state"}), 200
    
    df = pd.read_csv(DATA_PATH)
    df['Persona'] = df['Cluster'].map(persona_map)
    
    # Ensure all required columns are present
    required_cols = ['Recency', 'Frequency', 'Monetary', 'Cluster', 'Persona', 'CustomerID']
    if not all(col in df.columns for col in required_cols):
        return jsonify({"error": "Segmented data is missing required columns."}), 500
        
    return jsonify({
        "data": df[required_cols].to_dict(orient='records'),
        "persona_map": {str(k): v for k, v in persona_map.items()}
    })

@app.route('/metrics')
def get_metrics():
    if not os.path.exists(DATA_PATH):
        return jsonify({
            "total_customers": 0,
            "total_revenue": 0,
            "avg_recency": 0,
            "avg_frequency": 0,
            "avg_monetary": 0
        })
    
    df = pd.read_csv(DATA_PATH)
    metrics = {
        "total_customers": int(df['CustomerID'].nunique()),
        "total_revenue": float(df['Monetary'].sum()),
        "avg_recency": float(df['Recency'].mean()),
        "avg_frequency": float(df['Frequency'].mean()),
        "avg_monetary": float(df['Monetary'].mean())
    }
    return jsonify(metrics)

@app.route('/predict', methods=['POST'])
def predict_persona():
    if not scaler or not model or not persona_map:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        data = request.json
        recency = data['recency']
        frequency = data['frequency']
        monetary = data['monetary']
        
        # Create DataFrame for prediction
        input_df = pd.DataFrame([[recency, frequency, monetary]], columns=['Recency', 'Frequency', 'Monetary'])
        
        # Log transform
        input_log = np.log1p(input_df)
        
        # Scale
        input_scaled = scaler.transform(input_log)
        
        # Predict
        cluster = model.predict(input_scaled)[0]
        persona = persona_map.get(cluster, "Unknown")
        
        return jsonify({"persona": persona})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/predict_churn', methods=['POST'])
def predict_churn():
    if not churn_model or not churn_scaler:
        return jsonify({"error": "Churn model not loaded"}), 500
    
    try:
        data = request.json
        recency = data['recency']
        frequency = data['frequency']
        monetary = data['monetary']
        
        # Create DataFrame for prediction
        input_df = pd.DataFrame([[recency, frequency, monetary]], columns=['Recency', 'Frequency', 'Monetary'])
        
        # Scale
        input_scaled = churn_scaler.transform(input_df)
        
        # Predict
        prediction = churn_model.predict(input_scaled)[0]
        probability = churn_model.predict_proba(input_scaled)[0][1] # Probability of churn
        
        return jsonify({
            "churn_prediction": int(prediction),
            "churn_probability": float(probability)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/predict_ltv', methods=['POST'])
def predict_ltv():
    if not bgnbd_model or not ggf_model:
        return jsonify({"error": "LTV models not loaded"}), 500
    
    try:
        data = request.json
        frequency = data['frequency']
        recency = data['recency']
        T = data['T'] # T is the age of the customer
        monetary_value = data['monetary_value']

        # The models expect a pandas Series or array
        customer_data = pd.DataFrame([{
            'frequency': frequency,
            'recency': recency,
            'T': T,
            'monetary_value': monetary_value
        }])

        # Predict LTV for the next 12 months (365 days)
        ltv = ggf_model.customer_lifetime_value(
            bgnbd_model,
            customer_data['frequency'],
            customer_data['recency'],
            customer_data['T'],
            customer_data['monetary_value'],
            time=12,  # 12 months
            freq='D', # Daily frequency of transactions
            discount_rate=0.01 # Monthly discount rate
        )
        
        return jsonify({
            "predicted_ltv": float(ltv.iloc[0])
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/customer/<int:customer_id>')
def customer_profile(customer_id):
    """
    Displays a detailed profile for a single customer.
    """
    # --- 1. Load Data ---
    try:
        rfm_df = pd.read_csv(DATA_PATH)
        raw_df = pd.read_csv('data/online_retail.csv', encoding='ISO-8859-1')
    except FileNotFoundError:
        return "Data files not found. Please run the training pipeline.", 404

    # --- 2. Find Customer Data ---
    customer_rows = rfm_df[rfm_df['CustomerID'] == customer_id]
    if customer_rows.empty:
        return "Customer not found.", 404
    customer_rfm = customer_rows.iloc[0]

    # --- 3. Get Persona ---
    persona = persona_map.get(customer_rfm['Cluster'], "Unknown")

    # --- 4. Predict Churn ---
    churn_risk = "Unknown"
    churn_prob_percent = 0
    if churn_model and churn_scaler:
        customer_features = pd.DataFrame([customer_rfm[['Recency', 'Frequency', 'Monetary']]])
        scaled_features = churn_scaler.transform(customer_features)
        prediction = churn_model.predict(scaled_features)[0]
        probability = churn_model.predict_proba(scaled_features)[0][1]
        churn_risk = "High" if prediction == 1 else "Low"
        churn_prob_percent = round(probability * 100, 1)

    # --- 5. Predict LTV ---
    predicted_ltv = 0
    if bgnbd_model and ggf_model:
        # We need to calculate the summary data for this specific customer
        raw_df['InvoiceDate'] = pd.to_datetime(raw_df['InvoiceDate'])
        raw_df['TotalPrice'] = raw_df['Quantity'] * raw_df['UnitPrice']
        
        customer_transactions = raw_df[raw_df['CustomerID'] == customer_id]
        
        summary = summary_data_from_transaction_data(
            customer_transactions,
            customer_id_col='CustomerID',
            datetime_col='InvoiceDate',
            monetary_value_col='TotalPrice',
            observation_period_end=raw_df['InvoiceDate'].max()
        )
        
        if not summary.empty:
            predicted_ltv = ggf_model.customer_lifetime_value(
                bgnbd_model,
                summary['frequency'],
                summary['recency'],
                summary['T'],
                summary['monetary_value'],
                time=12, freq='D', discount_rate=0.01
            ).iloc[0]

    # --- 6. Get Transaction History ---
    transactions = raw_df[raw_df['CustomerID'] == customer_id].sort_values(by='InvoiceDate', ascending=False).to_dict(orient='records')

    # --- 7. Render Template ---
    return render_template(
        'customer_profile.html',
        customer_id=customer_id,
        persona=persona,
        churn_risk=churn_risk,
        churn_probability=churn_prob_percent,
        predicted_ltv=f"{predicted_ltv:,.2f}",
        transactions=transactions
    )

if __name__ == '__main__':
    app.run(debug=True, port=5001)
