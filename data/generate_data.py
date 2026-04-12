import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_synthetic_data(num_rows=5000):
    np.random.seed(42)
    
    # Generate Customer IDs
    customer_ids = np.random.randint(10000, 15000, size=num_rows)
    
    # Generate Invoice Dates (last 1 year)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    invoice_dates = [start_date + timedelta(seconds=np.random.randint(0, int((end_date - start_date).total_seconds()))) for _ in range(num_rows)]
    
    # Generate Quantities and Unit Prices
    quantities = np.random.randint(1, 20, size=num_rows)
    unit_prices = np.round(np.random.uniform(1.0, 50.0, size=num_rows), 2)
    
    # Generate Stock Codes and Descriptions (simplified)
    stock_codes = [f"STK{np.random.randint(100, 999)}" for _ in range(num_rows)]
    countries = np.random.choice(['United Kingdom', 'Germany', 'France', 'Spain', 'Italy'], size=num_rows)
    
    invoice_nos = [str(np.random.randint(536365, 581587)) for _ in range(num_rows)]

    df = pd.DataFrame({
        'InvoiceNo': invoice_nos,
        'StockCode': stock_codes,
        'Description': ['Item ' + s for s in stock_codes],
        'Quantity': quantities,
        'InvoiceDate': invoice_dates,
        'UnitPrice': unit_prices,
        'CustomerID': customer_ids,
        'Country': countries
    })
    
    # Ensure directory exists
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/online_retail.csv', index=False)
    print(f"Generated {num_rows} rows of synthetic data in data/online_retail.csv")

if __name__ == "__main__":
    generate_synthetic_data()
