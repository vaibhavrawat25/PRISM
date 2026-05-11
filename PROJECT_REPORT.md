# PRISM Project Report

## Title

PRISM: Customer Segmentation and Retention Dashboard Using RFM Analysis

## Objective

The objective of PRISM is to build a practical customer analytics dashboard that helps a retail business segment customers, identify valuable users, and estimate retention risk from transaction history.

## Motivation

Many small and medium businesses collect sales data but do not use it for customer decision-making. This project shows how machine learning can convert invoice-level records into useful customer groups for marketing, retention, and business analysis.

## Dataset

The project uses retail transaction data with invoice number, invoice date, quantity, unit price, customer ID, and product details. The app accepts CSV uploads and also includes a sample file for testing.

Required fields:

- InvoiceNo
- InvoiceDate
- Quantity
- UnitPrice
- CustomerID

## Methodology

1. Data Cleaning
   Invalid dates, missing customer IDs, and non-positive purchase rows are removed.

2. Feature Engineering
   Transaction data is aggregated into RFM features:
   - Recency: days since last purchase
   - Frequency: number of unique invoices
   - Monetary: total purchase value

3. Data Transformation
   RFM values are transformed using `log1p` to reduce skew and then scaled using `StandardScaler`.

4. Customer Segmentation
   K-Means clustering groups customers into up to four customer segments.

5. Persona Assignment
   Clusters are ranked using frequency, monetary value, and recency. The resulting personas are:
   - Champions
   - Potential Loyalists
   - At-Risk Customers
   - Lost Customers

6. Prediction Tools
   The dashboard includes manual RFM-based persona prediction and churn risk prediction.

## System Architecture

- Flask handles API routes, CSV upload, model inference, and page rendering.
- Pandas and NumPy handle data cleaning and RFM calculations.
- Scikit-learn handles scaling, clustering, and churn model inference.
- Plotly.js renders interactive charts.
- HTML, CSS, and JavaScript provide the dashboard interface.

## Implemented Modules

- CSV upload and validation
- RFM processing pipeline
- K-Means model persistence
- Segment dashboard
- Searchable customer table
- Persona prediction API
- Churn prediction API
- Customer profile view
- Sample CSV download

## Results

The app successfully converts transaction data into customer-level insights. It displays overall metrics, segment distribution, average revenue by segment, customer-level records, and individual profile details.

## Limitations

- The model depends on clean transaction data.
- K-Means clustering is unsupervised, so cluster labels are assigned using business logic.
- Churn prediction should be treated as an estimated retention signal unless trained with verified business churn labels.
- The current project uses file-based persistence instead of a production database.

## Future Enhancements

- Add authentication.
- Store uploaded datasets in PostgreSQL or MongoDB.
- Add model evaluation reports.
- Add campaign recommendation rules.
- Add deployment pipeline.
- Add unit tests and API tests.

## Conclusion

PRISM demonstrates a complete internship-level machine learning application: data ingestion, preprocessing, model usage, visual analytics, and a usable web interface. It is suitable for explaining practical ML, backend API design, and frontend dashboard development in an internship review.
