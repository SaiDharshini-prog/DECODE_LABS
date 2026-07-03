import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

from imblearn.over_sampling import SMOTE

# ==============================
# Load Dataset
# ==============================

df = pd.read_csv("creditcard.csv")

print("First 5 Rows")
print(df.head())

print("\nShape:", df.shape)

print("\nMissing Values")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())

df.drop_duplicates(inplace=True)

# ==============================
# Features and Target
# ==============================

X = df.drop("Class", axis=1)
y = df["Class"]

# ==============================
# Train Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==============================
# Feature Scaling
# ==============================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==============================
# SMOTE
# ==============================

print("\nApplying SMOTE...")

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

print("SMOTE Completed")

# ==============================
# Logistic Regression
# ==============================

print("\nTraining Logistic Regression...")

lr = LogisticRegression(
    max_iter=1000,
    random_state=42
)

lr.fit(X_train_smote, y_train_smote)

lr_pred = lr.predict(X_test)
lr_prob = lr.predict_proba(X_test)[:,1]

print("\n===== Logistic Regression =====")

print("Accuracy :", accuracy_score(y_test, lr_pred))
print("Precision:", precision_score(y_test, lr_pred))
print("Recall   :", recall_score(y_test, lr_pred))
print("F1 Score :", f1_score(y_test, lr_pred))
print("ROC AUC  :", roc_auc_score(y_test, lr_prob))

print("\nConfusion Matrix")

print(confusion_matrix(y_test, lr_pred))

print("\nClassification Report")

print(classification_report(y_test, lr_pred))

# ==============================
# Random Forest (FAST)
# ==============================

print("\nTraining Random Forest...")

rf = RandomForestClassifier(
    n_estimators=20,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train_smote, y_train_smote)

rf_pred = rf.predict(X_test)
rf_prob = rf.predict_proba(X_test)[:,1]

print("\n===== Random Forest =====")

print("Accuracy :", accuracy_score(y_test, rf_pred))
print("Precision:", precision_score(y_test, rf_pred))
print("Recall   :", recall_score(y_test, rf_pred))
print("F1 Score :", f1_score(y_test, rf_pred))
print("ROC AUC  :", roc_auc_score(y_test, rf_prob))

print("\nConfusion Matrix")

print(confusion_matrix(y_test, rf_pred))

print("\nClassification Report")

print(classification_report(y_test, rf_pred))

# ==============================
# Model Comparison
# ==============================

comparison = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest"],
    "Accuracy": [
        accuracy_score(y_test, lr_pred),
        accuracy_score(y_test, rf_pred)
    ],
    "Precision": [
        precision_score(y_test, lr_pred),
        precision_score(y_test, rf_pred)
    ],
    "Recall": [
        recall_score(y_test, lr_pred),
        recall_score(y_test, rf_pred)
    ],
    "F1 Score": [
        f1_score(y_test, lr_pred),
        f1_score(y_test, rf_pred)
    ],
    "ROC AUC": [
        roc_auc_score(y_test, lr_prob),
        roc_auc_score(y_test, rf_prob)
    ]
})

comparison.to_csv("Model_Comparison.csv", index=False)

print("\nModel_Comparison.csv saved successfully!")

print("\n================================")
print("PROJECT 2 COMPLETED SUCCESSFULLY")
print("================================")
