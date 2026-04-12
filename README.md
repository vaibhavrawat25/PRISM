# PRISM | Intelligence Engine 💎
### Advanced Behavioral Customer Segmentation Platform

---

## 📌 Vision
**PRISM** is a state-of-the-art Behavioral Intelligence platform designed to transform raw transaction data into actionable business personas. Unlike traditional segmentation scripts, PRISM utilizes a multi-stage Machine Learning pipeline—incorporating **Non-Linear Log Transformations**, **Standardized Feature Scaling**, and **Heuristic Personality Mapping**—to deliver industrial-grade customer insights through a premium Glassmorphism-themed dashboard.

## 🚀 Key Innovation: "Behavioral IQ"
PRISM goes beyond simple K-Means clustering by implementing a dual-metric performance ranking:
- **Champions**: Identified by **Lifecycle Value** (Frequency × Monetary). Your core revenue drivers.
- **Big Spenders**: Identified by **Ticket Size** (Monetary ÷ Frequency). High-value individuals who shop rare but large.
- **Potential Loafers**: Trial users or low-intent browsers requiring ticket-size incentives.
- **About to Sleep**: High-Recency dormant customers targeted for automated win-back campaigns.
- **Heuristic IQ Overlay**: A specialized post-processing layer that ensures 100% persona accuracy even on sparse or custom-uploaded datasets.

## 🛠️ Tech Stack
- **AI/ML Core**: Python, Scikit-learn (K-Means++), Pandas, NumPy.
- **The Brain**: Flask API with Joblib model persistence.
- **Visual Intelligence**: Plotly.js (High-Dimensional 3D Geometry), Lucide Icons.
- **Interface**: HTML5, Vanilla CSS3 (Glassmorphism & Dark Mode), ES6+ JavaScript.

## 📊 ML Pipeline Architecture
1. **Dynamic Ingestion**: Support for custom CSV uploads with automated RFM (Recency, Frequency, Monetary) calculation.
2. **Mathematical Normalization**: Implementation of `np.log1p()` transformation to handle skewness in frequency and monetary distributions.
3. **K-Means++ Optimization**: Unsupervised clustering using the Elbow-optimized centroids for high-dimensional grouping.
4. **Dynamic Mapping**: Performance ranking algorithm that automatically labels clusters based on their behavioral signals.

## 📂 Project Structure
```bash
PRISM/
├── backend/
│   ├── app.py          # Master Intelligence Engine (Flask API)
│   ├── model.pkl       # Trained K-Means Cluster Model
│   ├── scaler.pkl      # Feature Normalization Scaler
│   └── persona_map.pkl # Dynamic Behavioral Taxonomy
├── data/
│   ├── online_retail.csv   # Primary UCI Transactional Dataset
│   └── user_test.csv       # Custom Behavioral Test Cases
├── static/
│   ├── dashboard.js    # Reactive UI Logic & Plotly Visuals
│   ├── style.css       # Glassmorphism Design System
│   └── logo.png        # High-Tech Geometric Branding
├── templates/
│   └── index.html      # Premium Intelligence Dashboard
└── requirements.txt    # Industrial Dependencies
```

## ⚙️ Quick Start (Development)
1. **Initialize Environment**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Launch Engine**:
   ```bash
   python backend/app.py
   ```
3. **Analyze**:
   Open `http://localhost:5001` and upload your transaction CSV.

---

## 👨‍💻 Developer
**Vaibhav Rawat**  
B.Tech in Computer Science & Engineering  
**Specialization**: Machine Learning & Full-Stack Development | Batch 2026
