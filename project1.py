import pandas as pd
import numpy as np

# -----------------------------
# Read the Dataset
# -----------------------------
df = pd.read_excel("Dataset for Data Analytics.xlsx")

# -----------------------------
# 1. Display Basic Information
# -----------------------------
print("\n========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== SHAPE OF DATASET ==========")
print(df.shape)

print("\n========== COLUMN NAMES ==========")
print(df.columns)

print("\n========== DATASET INFORMATION ==========")
print(df.info())

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe(include='all'))

print("\n========== UNIQUE VALUES ==========")
print(df.nunique())

# -----------------------------
# 2. Check Missing Values
# -----------------------------
print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# Fill missing values
for column in df.columns:

    # For object (text) columns
    if df[column].dtype == "object":
        if df[column].isnull().sum() > 0:
            df[column] = df[column].fillna(df[column].mode()[0])

    # For numerical columns
    else:
        if df[column].isnull().sum() > 0:
            df[column] = df[column].fillna(df[column].median())

print("\n========== MISSING VALUES AFTER FILLING ==========")
print(df.isnull().sum())

# -----------------------------
# 3. Check Duplicate Rows
# -----------------------------
print("\n========== DUPLICATE ROWS ==========")
print(df.duplicated().sum())

df.drop_duplicates(inplace=True)

print("\nDuplicates After Removal:")
print(df.duplicated().sum())

# -----------------------------
# 4. Detect and Remove Outliers
# -----------------------------
numeric_columns = df.select_dtypes(include=np.number).columns

for column in numeric_columns:

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df = df[(df[column] >= lower) & (df[column] <= upper)]

print("\n========== SHAPE AFTER REMOVING OUTLIERS ==========")
print(df.shape)

# -----------------------------
# 5. Feature Engineering
# -----------------------------

# Convert Date column into datetime
df["Date"] = pd.to_datetime(df["Date"])

# Feature 1
df["Month"] = df["Date"].dt.month

# Feature 2
df["Day"] = df["Date"].dt.day

# Feature 3
df["AveragePricePerItem"] = df["TotalPrice"] / df["Quantity"]

# Feature 4
df["CouponUsed"] = np.where(df["CouponCode"].isnull(), "No", "Yes")

# -----------------------------
# 6. Correlation Matrix
# -----------------------------
print("\n========== CORRELATION MATRIX ==========")
print(df.select_dtypes(include=np.number).corr())

# -----------------------------
# 7. Final Dataset Information
# -----------------------------
print("\n========== FINAL DATASET SHAPE ==========")
print(df.shape)

print("\n========== FINAL COLUMN NAMES ==========")
print(df.columns)

print("\n========== FIRST 5 ROWS OF CLEANED DATA ==========")
print(df.head())

# -----------------------------
# 8. Save Cleaned Dataset
# -----------------------------
df.to_excel("Cleaned_Dataset.xlsx", index=False)

df.to_csv("Cleaned_Dataset.csv", index=False)

print("\n========================================")
print("PROJECT COMPLETED SUCCESSFULLY")
print("========================================")
print("Cleaned Excel File  : Cleaned_Dataset.xlsx")
print("Cleaned CSV File    : Cleaned_Dataset.csv")
print("Final Dataset Shape :", df.shape)