from src.anomaly_detector import run_anomaly_detection
from src.data_generator import generate_sample_data
from src.preprocess import preprocess_data
from src.report_generator import generate_reports
from src.train_model import train_models


def main():
    print("Starting AI-Powered Industrial Safety Intelligence project...")

    print("\n1. Generating synthetic industrial safety data")
    data_info = generate_sample_data(row_count=5000)
    print(data_info)

    print("\n2. Cleaning and preparing final dataset")
    processed_path = preprocess_data()
    print(f"Processed dataset saved at: {processed_path}")

    print("\n3. Training AI risk prediction models")
    best_model, scores = train_models()
    print(scores)
    print(f"Best model selected: {best_model}")

    print("\n4. Running anomaly detection")
    anomaly_path = run_anomaly_detection()
    print(f"Anomaly report saved at: {anomaly_path}")

    print("\n5. Generating safety reports")
    report_paths = generate_reports()
    print(report_paths)

    print("\nProject run completed. You can now start the dashboard:")
    print("streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
