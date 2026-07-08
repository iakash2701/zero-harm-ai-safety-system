# AI-Powered Industrial Safety Intelligence for Zero-Harm Operations

This is a hackathon prototype built to show how AI can help safety officers detect industrial risks before accidents happen.

This project was developed as a prototype to demonstrate how AI can combine sensor data, worker location, maintenance records, and safety rules to predict industrial risks before accidents happen.

## Project Links

- Vercel Demo: [https://industrial-safety-ai.vercel.app](https://industrial-safety-ai.vercel.app)
- GitHub Repository: [https://github.com/iakash2701/zero-harm-ai-safety-system](https://github.com/iakash2701/zero-harm-ai-safety-system)

## Problem Statement

In many industries, safety systems work separately. IoT sensors, gas detectors, worker attendance, work permits, CCTV, SCADA, and maintenance logs may all generate alerts, but they are not always connected. Because of this, a safety officer may miss compound risks like high gas level plus worker entry plus invalid permit.

## Proposed Solution

The project creates a small AI-based safety intelligence system. It generates realistic synthetic industrial data, cleans and merges the data, trains ML models to classify safety risk, detects unusual sensor readings, applies rule-based compound risk checks, and gives practical recommendations.

Synthetic data is used because real industrial safety data is difficult to access and usually contains private worker, machine, and incident information.

## Main Features

- Synthetic sensor, worker, maintenance, permit, and incident datasets
- Risk labels: Low, Medium, High, Critical
- Data preprocessing and missing value handling
- Logistic Regression, Random Forest, and Gradient Boosting model comparison
- Automatic best model selection using weighted F1-score
- Isolation Forest anomaly detection
- Compound risk engine for combined dangerous situations
- Safety recommendations for each alert
- Daily CSV and PDF safety reports
- Streamlit dashboard for live demo presentation

## Tech Stack

Python, Pandas, NumPy, Scikit-learn, Streamlit, Plotly, Joblib, ReportLab

## Folder Structure

```text
industrial-safety-ai/
├── data/
│   ├── raw/
│   ├── processed/
│   └── sample/
├── models/
├── notebooks/
├── src/
│   ├── data_generator.py
│   ├── preprocess.py
│   ├── train_model.py
│   ├── risk_engine.py
│   ├── anomaly_detector.py
│   ├── recommendation_engine.py
│   ├── report_generator.py
│   └── utils.py
├── dashboard/
│   └── app.py
├── reports/
├── requirements.txt
├── README.md
└── run_project.py
```

## How To Install

```bash
pip install -r requirements.txt
```

## How To Run

Run the complete pipeline:

```bash
python run_project.py
```

Start the dashboard:

```bash
streamlit run dashboard/app.py
```

## Dataset Details

The project automatically creates these files inside `data/sample/`:

- `sensor_data.csv`
- `worker_location.csv`
- `maintenance_logs.csv`
- `work_permits.csv`
- `incident_logs.csv`

The sensor dataset has at least 5,000 records. Values are generated with practical ranges for temperature, gas level, vibration, pressure, and noise. Some missing values are added on purpose, then cleaned during preprocessing.

Example zones include Zone-A, Zone-B, Zone-C, Furnace Area, Chemical Storage, and Assembly Point.

## Model Details

Three simple models are trained:

- Logistic Regression
- Random Forest
- Gradient Boosting

The models are compared using accuracy, precision, recall, F1-score, and confusion matrix. The best model is selected using weighted F1-score and saved as `models/risk_prediction_model.pkl`.

In the generated sample run, Gradient Boosting performed slightly better than Random Forest. This makes sense for the prototype because the risk labels come from many small rule combinations, and boosting can learn those step-by-step patterns well. Random Forest also performs strongly because it handles mixed sensor and categorical safety features better than a simple linear model.

The results are kept realistic for a synthetic hackathon prototype. The project does not claim perfect or 99% accuracy.

## Dashboard Pages

- Overview
- Live Sensor Monitoring
- AI Risk Prediction
- Compound Risk Alerts
- Worker Location Map
- Maintenance Status
- Work Permit Status
- Safety Recommendations
- Reports

The dashboard shows KPI cards, gas and temperature trends, vibration trends, risk distribution, zone-wise alerts, latest alerts, manual risk prediction, map view, and generated reports.

## Reports

The report generator creates:

- `reports/daily_summary.csv`
- `reports/daily_safety_alerts.csv`
- `reports/anomaly_report.csv`
- `reports/model_comparison.csv`
- `reports/daily_safety_report.pdf`

Reports include alerts, reasons, risk score, recommended actions, worker violations, maintenance issues, and permit violations.

## Limitations

This prototype uses synthetic data and rule-based assumptions. For real-world deployment, the model must be trained and validated using actual industrial sensor and incident data.

The dashboard is made for demo and presentation. It does not connect to real IoT devices, CCTV, SCADA, emergency alarms, or company databases yet.

## Future Improvements

- Integration with real IoT sensors, CCTV cameras, SCADA systems, emergency alarms, and company safety databases
- Real-time streaming using MQTT or Kafka
- Better geospatial alerts for restricted zones
- Worker mobile app notifications
- More detailed incident investigation workflow
- Role-based login for safety officers and supervisors
- Model retraining using real incident feedback

## Project Note

This project is made in a practical student-hackathon style. The code is kept simple so that a B.Tech AI & ML team can understand it, modify it, and explain it during a demo.
