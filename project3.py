import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ===============================
# Load Dataset
# ===============================

df = pd.read_excel("Dataset for Data Analytics.xlsx")

print("First 5 Rows")
print(df.head())

print("\nDataset Shape")
print(df.shape)

print("\nMissing Values")
print(df.isnull().sum())

# ===============================
# Customer Segmentation Dataset
# ===============================

customer_df = df.groupby("CustomerID").agg(
    TotalQuantity=("Quantity", "sum"),
    AvgItems=("ItemsInCart", "mean"),
    AvgUnitPrice=("UnitPrice", "mean"),
    TotalSpent=("TotalPrice", "sum")
).reset_index()

print("\nCustomer Data")
print(customer_df.head())

# ===============================
# Feature Selection
# ===============================

X = customer_df[
    ["TotalQuantity",
     "AvgItems",
     "AvgUnitPrice",
     "TotalSpent"]
]

# ===============================
# Standardization
# ===============================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ===============================
# PCA
# ===============================

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

print("\nExplained Variance Ratio")

print(pca.explained_variance_ratio_)

# ===============================
# Elbow Method
# ===============================

wcss = []

for k in range(2,11):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X_pca)

    wcss.append(model.inertia_)

plt.figure(figsize=(8,5))

plt.plot(range(2,11), wcss, marker='o')

plt.title("Elbow Method")

plt.xlabel("Number of Clusters")

plt.ylabel("WCSS")

plt.savefig("Elbow_Method.png")

plt.close()

print("\nElbow graph saved.")

# ===============================
# Silhouette Score
# ===============================

best_score = -1

best_k = 2

for k in range(2,11):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_pca)

    score = silhouette_score(
        X_pca,
        labels
    )

    print("K =", k,
          "Silhouette Score =", score)

    if score > best_score:

        best_score = score

        best_k = k

print("\nBest Number of Clusters =", best_k)

# ===============================
# Final KMeans
# ===============================

kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

customer_df["Cluster"] = kmeans.fit_predict(X_pca)

# ===============================
# Customer Personas
# ===============================

persona = {
    0: "Budget Customers",
    1: "Regular Customers",
    2: "Premium Customers",
    3: "High Value Customers",
    4: "VIP Customers"
}

customer_df["Persona"] = customer_df["Cluster"].map(
    lambda x: persona.get(x, "Customer")
)

# ===============================
# Cluster Summary
# ===============================

print("\nCluster Summary")

print(customer_df.groupby("Cluster").mean(numeric_only=True))

# ===============================
# Cluster Plot
# ===============================

plt.figure(figsize=(8,6))

plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=customer_df["Cluster"]
)

plt.title("Customer Segmentation")

plt.xlabel("PCA 1")

plt.ylabel("PCA 2")

plt.savefig("Customer_Clusters.png")

plt.close()

print("\nCluster graph saved.")

# ===============================
# Save Output
# ===============================

customer_df.to_excel(
    "Customer_Segmentation_Output.xlsx",
    index=False
)

customer_df.to_csv(
    "Customer_Segmentation_Output.csv",
    index=False
)

print("\nProject 3 Completed Successfully!")

print("\nFiles Generated")

print("1. Elbow_Method.png")

print("2. Customer_Clusters.png")

print("3. Customer_Segmentation_Output.xlsx")

print("4. Customer_Segmentation_Output.csv")