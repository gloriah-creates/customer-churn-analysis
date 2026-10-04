import pandas as pd
import numpy as np


# ============================================================
# CUSTOMER CHURN ANALYSIS
# Stage 1: Data Audit
# ============================================================

# ------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------

DATA_PATH = "../data/Telco-Customer-Churn.csv"

df = pd.read_csv(DATA_PATH)


# ------------------------------------------------------------
# 2. Basic dataset information
# ------------------------------------------------------------

print("=" * 70)
print("CUSTOMER CHURN DATA AUDIT")
print("=" * 70)

print("\n[1] DATASET SHAPE")
print("-" * 70)
print(f"Rows:    {df.shape[0]:,}")
print(f"Columns: {df.shape[1]:,}")


# ------------------------------------------------------------
# 3. Column names
# ------------------------------------------------------------

print("\n[2] COLUMN NAMES")
print("-" * 70)

for i, column in enumerate(df.columns, start=1):
    print(f"{i:2}. {column}")


# ------------------------------------------------------------
# 4. Data types
# ------------------------------------------------------------

print("\n[3] DATA TYPES")
print("-" * 70)

print(df.dtypes)


# ------------------------------------------------------------
# 5. Missing values
# ------------------------------------------------------------

print("\n[4] MISSING VALUES")
print("-" * 70)

missing = df.isnull().sum()

missing_table = pd.DataFrame({
    "missing_count": missing,
    "missing_percent": (missing / len(df) * 100).round(2)
})

missing_table = missing_table[
    missing_table["missing_count"] > 0
].sort_values(
    "missing_count",
    ascending=False
)

if missing_table.empty:
    print("No standard missing values detected.")
else:
    print(missing_table)


# ------------------------------------------------------------
# 6. Duplicate rows
# ------------------------------------------------------------

print("\n[5] DUPLICATE ROWS")
print("-" * 70)

duplicate_count = df.duplicated().sum()

print(f"Duplicate rows: {duplicate_count:,}")


# ------------------------------------------------------------
# 7. Unique values
# ------------------------------------------------------------

print("\n[6] UNIQUE VALUES BY COLUMN")
print("-" * 70)

unique_table = pd.DataFrame({
    "column": df.columns,
    "unique_values": [df[col].nunique(dropna=False) for col in df.columns]
})

print(unique_table.to_string(index=False))


# ------------------------------------------------------------
# 8. Target variable: Churn
# ------------------------------------------------------------

print("\n[7] CHURN DISTRIBUTION")
print("-" * 70)

churn_counts = df["Churn"].value_counts()

churn_percentages = (
    df["Churn"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

churn_table = pd.DataFrame({
    "customers": churn_counts,
    "percentage": churn_percentages
})

print(churn_table)


# ------------------------------------------------------------
# 9. Check TotalCharges
# ------------------------------------------------------------

print("\n[8] TOTALCHARGES DATA QUALITY CHECK")
print("-" * 70)

print(f"Current data type: {df['TotalCharges'].dtype}")

total_charges_numeric = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

invalid_total_charges = total_charges_numeric.isna().sum()

print(
    f"Values that cannot be converted to numeric: "
    f"{invalid_total_charges:,}"
)

if invalid_total_charges > 0:

    print("\nRows with non-numeric TotalCharges:")
    print(
        df.loc[
            total_charges_numeric.isna(),
            [
                "customerID",
                "tenure",
                "TotalCharges",
                "Churn"
            ]
        ].to_string(index=False)
    )


# ------------------------------------------------------------
# 10. Numerical summary
# ------------------------------------------------------------

print("\n[9] NUMERICAL SUMMARY")
print("-" * 70)

print(df.describe().T)


# ------------------------------------------------------------
# 11. Categorical summary
# ------------------------------------------------------------

print("\n[10] CATEGORICAL VARIABLES")
print("-" * 70)

categorical_columns = df.select_dtypes(
    include=["object"]
).columns

for column in categorical_columns:

    print(f"\n--- {column} ---")

    print(
        df[column]
        .value_counts(dropna=False)
        .to_string()
    )


# ------------------------------------------------------------
# 12. Customer ID uniqueness
# ------------------------------------------------------------

print("\n[11] CUSTOMER ID CHECK")
print("-" * 70)

unique_customer_ids = df["customerID"].nunique()

print(f"Total rows:          {len(df):,}")
print(f"Unique customer IDs: {unique_customer_ids:,}")

if unique_customer_ids == len(df):
    print("Every row has a unique customer ID.")
else:
    print("WARNING: Duplicate customer IDs detected.")


# ------------------------------------------------------------
# 13. Display sample records
# ------------------------------------------------------------

print("\n[12] SAMPLE RECORDS")
print("-" * 70)

print(df.head().to_string(index=False))


# ------------------------------------------------------------
# 14. Final audit summary
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)

print(f"Rows:                 {len(df):,}")
print(f"Columns:              {len(df.columns):,}")
print(f"Duplicate rows:       {duplicate_count:,}")
print(f"Missing-value cells:  {df.isnull().sum().sum():,}")
print(f"Unique customers:     {unique_customer_ids:,}")
print(f"Invalid TotalCharges: {invalid_total_charges:,}")

print("\nNext stage: data cleaning and preparation.")
print("=" * 70)