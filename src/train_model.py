# ============================================================
# CUSTOMER CHURN MODEL TRAINING
# Save the complete preprocessing + Logistic Regression model
# ============================================================

import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression


# ============================================================
# 1. PATHS
# ============================================================

DATA_PATH = "Data/Telco-Customer-Churn.csv"
MODEL_DIR = "src"
MODEL_PATH = os.path.join(MODEL_DIR, "churn_model.pkl")


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 70)
print("CUSTOMER CHURN MODEL TRAINING")
print("=" * 70)

print("\n[1] LOADING DATA")
print("-" * 70)

data = pd.read_csv(DATA_PATH)

print(f"Dataset loaded successfully: {len(data):,} customers")


# ============================================================
# 3. CLEAN DATA
# ============================================================

print("\n[2] CLEANING DATA")
print("-" * 70)

# Convert TotalCharges to numeric.
# Invalid values become missing values.
data["TotalCharges"] = pd.to_numeric(
    data["TotalCharges"],
    errors="coerce"
)

# Fill missing TotalCharges using the median.
data["TotalCharges"] = data["TotalCharges"].fillna(
    data["TotalCharges"].median()
)

# Convert Churn into binary target.
data["Churn_binary"] = (
    data["Churn"]
    .map({
        "No": 0,
        "Yes": 1
    })
)

print(
    f"Missing TotalCharges: "
    f"{data['TotalCharges'].isna().sum()}"
)

print(
    f"Missing Churn_binary: "
    f"{data['Churn_binary'].isna().sum()}"
)


# ============================================================
# 4. DEFINE FEATURES AND TARGET
# ============================================================

print("\n[3] PREPARING FEATURES")
print("-" * 70)

numerical_features = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

categorical_features = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod"
]

feature_columns = (
    numerical_features +
    categorical_features
)

X = data[feature_columns]
y = data["Churn_binary"]


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

print("\n[4] TRAIN / TEST SPLIT")
print("-" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Training customers: {len(X_train):,}")
print(f"Testing customers:  {len(X_test):,}")


# ============================================================
# 6. BUILD PREPROCESSING PIPELINE
# ============================================================

print("\n[5] BUILDING PREPROCESSING PIPELINE")
print("-" * 70)

numerical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            numerical_pipeline,
            numerical_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ============================================================
# 7. CREATE FINAL MODEL
# ============================================================

print("\n[6] TRAINING LOGISTIC REGRESSION")
print("-" * 70)

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            LogisticRegression(
                max_iter=2000,
                random_state=42
            )
        )
    ]
)

model.fit(
    X_train,
    y_train
)

print("Model trained successfully.")


# ============================================================
# 8. SAVE MODEL
# ============================================================

print("\n[7] SAVING MODEL")
print("-" * 70)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

joblib.dump(
    model,
    MODEL_PATH
)

print(
    f"Model saved successfully:\n"
    f"{MODEL_PATH}"
)


# ============================================================
# 9. VERIFY MODEL
# ============================================================

print("\n[8] MODEL VERIFICATION")
print("-" * 70)

test_predictions = model.predict(X_test)

test_probabilities = model.predict_proba(
    X_test
)[:, 1]

print(
    f"Predictions generated: "
    f"{len(test_predictions):,}"
)

print(
    f"Probability range: "
    f"{test_probabilities.min():.4f} - "
    f"{test_probabilities.max():.4f}"
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("MODEL TRAINING COMPLETE")
print("=" * 70)

print(
    "\nSaved model:"
    f"\n{MODEL_PATH}"
)

print(
    "\nThis file contains both:"
    "\n1. Data preprocessing"
    "\n2. Logistic Regression model"
)

print(
    "\nThe Streamlit application can now load "
    "this file and make predictions."
)

print("=" * 70)