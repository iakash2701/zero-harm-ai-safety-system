import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from src.utils import MODEL_DIR, PROCESSED_DIR, REPORT_DIR, make_project_dirs


FEATURE_COLUMNS = [
    "zone_id",
    "temperature",
    "gas_level",
    "vibration",
    "pressure",
    "noise_level",
    "machine_status",
    "ppe_status",
    "risk_zone",
    "entry_status",
    "fault_history_count",
    "maintenance_status",
    "permit_status",
    "maintenance_overdue",
    "ppe_violation",
    "invalid_permit",
    "restricted_entry",
]


def build_preprocessor():
    numeric_cols = [
        "temperature",
        "gas_level",
        "vibration",
        "pressure",
        "noise_level",
        "fault_history_count",
        "maintenance_overdue",
        "ppe_violation",
        "invalid_permit",
        "restricted_entry",
    ]
    category_cols = [
        "zone_id",
        "machine_status",
        "ppe_status",
        "risk_zone",
        "entry_status",
        "maintenance_status",
        "permit_status",
    ]

    return ColumnTransformer([
        ("numbers", StandardScaler(), numeric_cols),
        ("categories", OneHotEncoder(handle_unknown="ignore"), category_cols),
    ])


def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=800, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=120, random_state=42, class_weight="balanced"),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }


def train_models():
    make_project_dirs()
    data_path = PROCESSED_DIR / "final_safety_dataset.csv"
    df = pd.read_csv(data_path)

    x = df[FEATURE_COLUMNS]
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["risk_label"])

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    results = []
    trained_models = {}

    for model_name, model in get_models().items():
        pipeline = Pipeline([
            ("preprocess", build_preprocessor()),
            ("model", model),
        ])
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)

        row = {
            "model": model_name,
            "accuracy": round(accuracy_score(y_test, predictions), 3),
            "precision": round(precision_score(y_test, predictions, average="weighted", zero_division=0), 3),
            "recall": round(recall_score(y_test, predictions, average="weighted", zero_division=0), 3),
            "f1_score": round(f1_score(y_test, predictions, average="weighted", zero_division=0), 3),
        }
        results.append(row)
        trained_models[model_name] = {
            "pipeline": pipeline,
            "predictions": predictions,
        }

    results_df = pd.DataFrame(results).sort_values("f1_score", ascending=False)
    best_model_name = results_df.iloc[0]["model"]
    best_pipeline = trained_models[best_model_name]["pipeline"]
    best_predictions = trained_models[best_model_name]["predictions"]

    model_bundle = {
        "model": best_pipeline,
        "label_encoder": label_encoder,
        "feature_columns": FEATURE_COLUMNS,
        "best_model_name": best_model_name,
        "model_scores": results,
    }

    joblib.dump(model_bundle, MODEL_DIR / "risk_prediction_model.pkl")
    results_df.to_csv(REPORT_DIR / "model_comparison.csv", index=False)

    report = classification_report(
        y_test,
        best_predictions,
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0,
    )
    with open(REPORT_DIR / "classification_report.json", "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    matrix = confusion_matrix(y_test, best_predictions)
    matrix_df = pd.DataFrame(matrix, index=label_encoder.classes_, columns=label_encoder.classes_)
    matrix_df.to_csv(REPORT_DIR / "confusion_matrix.csv")

    return best_model_name, results_df


def predict_risk(input_data):
    bundle = joblib.load(MODEL_DIR / "risk_prediction_model.pkl")
    row = pd.DataFrame([input_data])
    row = row[bundle["feature_columns"]]
    encoded_prediction = bundle["model"].predict(row)[0]
    probabilities = bundle["model"].predict_proba(row)[0]
    label = bundle["label_encoder"].inverse_transform([encoded_prediction])[0]
    confidence = float(max(probabilities))
    return label, confidence


if __name__ == "__main__":
    print(train_models()[0])
