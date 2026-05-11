# PRISM - Customer Segmentation and Retention Dashboard

PRISM is a Flask-based machine learning dashboard that converts retail transaction data into customer segments using RFM analysis and K-Means clustering. The project is designed as an internship-level B.Tech CSE project with a practical business workflow: upload data, generate personas, inspect segment analytics, and predict customer-level persona or churn risk.

## Problem Statement

Retail businesses often have transaction data but no simple way to identify valuable, inactive, or at-risk customers. PRISM solves this by grouping customers based on:

- Recency: days since last purchase
- Frequency: number of unique invoices/orders
- Monetary: total purchase value

The dashboard helps identify customer personas that can support retention and marketing decisions.

## Key Features

- CSV upload for transaction datasets
- Automatic RFM feature engineering
- K-Means customer segmentation
- Dynamic persona assignment
- Segment distribution and revenue charts
- Searchable customer segment table
- Customer persona prediction from manual RFM values
- Churn risk prediction from RFM values
- Customer profile page with transaction history
- Sample CSV download for quick testing
- Light and dark theme UI

## Tech Stack

- Backend: Flask, Python
- Data processing: Pandas, NumPy
- Machine learning: Scikit-learn, Joblib
- CLV models: Lifetimes
- Frontend: HTML, CSS, JavaScript
- Charts: Plotly.js
- Icons: Lucide

## Dataset Format

Upload a CSV with these required transaction columns:

```csv
InvoiceNo,StockCode,Description,Quantity,InvoiceDate,UnitPrice,CustomerID,Country
582001,PR01,Product A,5,2025-12-17 10:00:00,60.20,15001,India
```

Minimum required columns:

- `InvoiceNo`
- `InvoiceDate`
- `Quantity`
- `UnitPrice`
- `CustomerID`

The app also supports precomputed RFM files with:

- `CustomerID`
- `Recency`
- `Frequency`
- `Monetary`

If the CSV is missing required columns, the upload API returns a clear error message such as `Missing required transaction columns: InvoiceDate, InvoiceNo`.

## Machine Learning Workflow

1. Clean invalid rows and missing customer IDs.
2. Convert invoice dates and numeric purchase fields.
3. Calculate `TotalPrice = Quantity * UnitPrice`.
4. Aggregate transactions into customer-level RFM features.
5. Apply `log1p` transformation to reduce skew.
6. Standardize features using `StandardScaler`.
7. Cluster customers using K-Means.
8. Assign personas based on frequency, monetary value, and recency score.

## Personas

- Champions: frequent, recent, high-value customers
- Potential Loyalists: engaged customers with growth potential
- At-Risk Customers: customers whose engagement is dropping
- Lost Customers: inactive customers with low recent value

## Churn Predictor Note

The churn predictor is useful for demonstration and learning, but its quality depends on how churn labels are created during training. In this project, churn should be interpreted as a model-based retention signal from RFM behavior, not a guaranteed business outcome. For production use, churn labels should be created from a clear business rule such as "no purchase for more than X days."

## Project Structure

```text
PRISM/
├── backend/
│   ├── app.py
│   ├── model.pkl
│   ├── scaler.pkl
│   ├── persona_map.pkl
│   ├── churn_model.pkl
│   └── churn_scaler.pkl
├── data/
│   ├── online_retail.csv
│   ├── rfm_segmented.csv
│   └── user_test.csv
├── static/
│   ├── dashboard.js
│   ├── style.css
│   └── logo.png
├── templates/
│   ├── index.html
│   └── customer_profile.html
├── run_training_pipeline.py
├── PROJECT_REPORT.md
└── requirements.txt
```

## How To Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the Flask app:

```bash
python backend/app.py
```

Open:

```text
http://127.0.0.1:5001
```

If port `5001` is already busy, run the app on another port:

```bash
python -c "import backend.app as prism; prism.app.run(debug=True, port=5002)"
```

## How To Test Upload

1. Open the app.
2. Click `Sample CSV` to download a test file.
3. Upload the CSV.
4. Check the dashboard metrics, charts, and customer segment table.
5. Try the `Customer Persona Predictor` and `Churn Risk Predictor`.

## Automated Tests

The project includes simple API tests using Python's built-in `unittest` module. These tests cover:

- uploading a valid CSV
- rejecting an invalid CSV
- checking that `/metrics` returns JSON
- checking that `/predict` returns a persona

Run tests:

```bash
python -m unittest discover -s tests
```

## Limitations

- The model uses only transaction history and does not include demographic, web, or campaign interaction data.
- K-Means requires manual interpretation of clusters.
- Churn prediction depends on the quality of training labels.
- Uploaded data should follow the required CSV schema.

## Future Scope

- Add login and role-based dashboard access.
- Store uploads and predictions in a database.
- Add model evaluation metrics and confusion matrix for churn.
- Add campaign recommendations by persona.
- Deploy the app on Render, Railway, or AWS.
- Add automated tests for upload validation and prediction APIs.

