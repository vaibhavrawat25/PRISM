from flask import Flask, render_template, jsonify, request, send_file
import pandas as pd
import joblib
import os
import numpy as np
import io
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

app = Flask(__name__, template_folder='../templates', static_folder='../static')

# Paths
MODEL_PATH = 'backend/model.pkl'
SCALER_PATH = 'backend/scaler.pkl'
MAP_PATH = 'backend/persona_map.pkl'
DATA_PATH = 'data/rfm_segmented.csv'

# Persistent objects
model = None
scaler = None
persona_map = None

def load_persistence():
    global model, scaler, persona_map
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
    if os.path.exists(MAP_PATH):
        persona_map = joblib.load(MAP_PATH)

load_persistence()

def process_rfm(df):
    """Elite ML pipeline with Behavioral Persona Mapping."""
    # Basic cleaning
    df = df.dropna(subset=['CustomerID'])
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
    df = df.dropna(subset=['InvoiceDate'])
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    
    # RFM Aggregation
    snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'InvoiceNo': 'count',
        'TotalPrice': 'sum'
    })
    rfm.rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'TotalPrice': 'Monetary'}, inplace=True)

    # 1. Log Transformation
    rfm_log = rfm.copy()
    rfm_log['Frequency'] = np.log1p(rfm_log['Frequency'])
    rfm_log['Monetary'] = np.log1p(rfm_log['Monetary'])
    
    # 2. Scaling
    scaler_new = StandardScaler()
    scaled_data = scaler_new.fit_transform(rfm_log)
    
    # 3. K-Means
    k = min(4, len(rfm))
    kmeans_new = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    rfm['Cluster'] = kmeans_new.fit_predict(scaled_data)
    
    # 4. 🔥 Elite Behavioral Mapping
    # Calculate persona characteristics
    cluster_stats = rfm.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean'
    })
    
    # Lifecycle Value (Frequency * Monetary)
    cluster_stats['Lifecycle'] = cluster_stats['Frequency'] * cluster_stats['Monetary']
    # Ticket Size (Monetary / Frequency)
    cluster_stats['TicketSize'] = cluster_stats['Monetary'] / cluster_stats['Frequency']
    
    new_map = {}
    
    # 1. Champions: Highest Lifecycle Value
    champ_idx = cluster_stats['Lifecycle'].idxmax()
    new_map[champ_idx] = "Champions"
    
    # 2. About to Sleep: Highest Recency (among remaining)
    remaining = cluster_stats.drop(champ_idx)
    sleep_idx = remaining['Recency'].idxmax()
    new_map[sleep_idx] = "About to Sleep"
    
    # 3. Big Spenders: Highest Ticket Size (among remaining)
    remaining = remaining.drop(sleep_idx)
    if not remaining.empty:
        spender_idx = remaining['TicketSize'].idxmax()
        new_map[spender_idx] = "Big Spenders"
        
        # 4. Potential Loafers: The final cluster
        remaining = remaining.drop(spender_idx)
        if not remaining.empty:
            loafer_idx = remaining.index[0]
            new_map[loafer_idx] = "Potential Loafers"
    
    # Persistence
    joblib.dump(kmeans_new, MODEL_PATH)
    joblib.dump(scaler_new, SCALER_PATH)
    joblib.dump(new_map, MAP_PATH)
    rfm.to_csv(DATA_PATH)
    load_persistence()
    return rfm

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/segment')
def get_segments():
    if not os.path.exists(DATA_PATH) or persona_map is None:
        return jsonify({"error": "empty_state"}), 200
    
    rfm = pd.read_csv(DATA_PATH)
    stats = {
        "total_customers": int(rfm.shape[0]),
        "avg_recency": float(rfm['Recency'].mean()),
        "avg_frequency": float(rfm['Frequency'].mean()),
        "avg_monetary": float(rfm['Monetary'].mean())
    }
    
    # Ensure persona_map keys are JSON-serializable strings
    serializable_map = {str(k): v for k, v in persona_map.items()}
    
    return jsonify({
        "stats": stats,
        "data": rfm.to_dict(orient='records'),
        "persona_map": serializable_map
    })

@app.route('/metrics')
def get_metrics():
    if not os.path.exists(DATA_PATH):
        return jsonify({"error": "No data found"}), 404
    
    rfm = pd.read_csv(DATA_PATH)
    metrics = rfm.groupby('Cluster').agg({
        'Recency': 'mean', 'Frequency': 'mean', 'Monetary': 'mean', 'CustomerID': 'count'
    }).reset_index()
    metrics.rename(columns={'CustomerID': 'Count'}, inplace=True)
    return jsonify(metrics.to_dict(orient='records'))

@app.route('/upload', methods=['POST'])
def upload_data():
    if 'file' not in request.files:
        return jsonify({"error": "Missing file"}), 400
    file = request.files['file']
    if not file.filename.endswith('.csv'):
        return jsonify({"error": "Invalid format. CSV required."}), 400
    
    try:
        df = pd.read_csv(file)
        process_rfm(df)
        return jsonify({"message": "PRISM IQ synchronized."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None or persona_map is None:
        return jsonify({"error": "PRISM Engine not ready."}), 400
    
    try:
        data = request.get_json()
        r, f, m = float(data['recency']), float(data['frequency']), float(data['monetary'])
        
        # Apply same ML pipeline
        f_log, m_log = np.log1p(f), np.log1p(m)
        scaled = scaler.transform([[r, f_log, m_log]])
        cluster = int(model.predict(scaled)[0])
        
        base_persona = persona_map.get(cluster, "Unknown")
        
        # 🔥 Heuristic Overlay for Sparse Data Tuning
        # If the model labels it as a top-performer, but it's a one-time huge spend
        if base_persona == "Champions" and f <= 2 and m > 1000:
            base_persona = "Big Spenders"
        
        return jsonify({
            "cluster": cluster,
            "persona": base_persona
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/export')
def export_data():
    if not os.path.exists(DATA_PATH) or persona_map is None:
        return "Error", 404
        
    rfm = pd.read_csv(DATA_PATH)
    # Map personas correctly for the export
    rfm['Persona'] = rfm['Cluster'].map(lambda x: persona_map.get(int(x), "Unknown"))
    
    # 🗄️ Excel/CSV Sanitization
    output = io.BytesIO()
    # Write BOM for Excel
    output.write(b'\xef\xbb\xbf')
    # Use utf-8 so sig is implicit
    csv_str = rfm.to_csv(index=False, encoding='utf-8')
    output.write(csv_str.encode('utf-8'))
    output.seek(0)
    
    return send_file(
        output,
        mimetype="text/csv",
        as_attachment=True,
        download_name="PRISM_Persona_Analysis.csv"
    )

if __name__ == '__main__':
    app.run(debug=True, port=5001)
