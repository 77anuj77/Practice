# assignment 4
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# SECTION-A
df = pd.read_csv(
    "/Users/ayushparoha/Documents/Data_Science/env/college/ass4/PS_20174392719_1491204439457_log.csv"
)
print(df.head())
print(df.info())
print(df.describe())
print(df.shape)
Numerical_features = df.select_dtypes(include=np.number).columns
print(Numerical_features)
# value_counts is a function used to detect the unique values in a colum at[0] then genunine transaction
frauds = df["isFraud"].value_counts()
print("Genunine Transaction: ", frauds[0])
print("Fraud Transaction", frauds[1])
percentage = (frauds[1] / len(df)) * 100
print("Fraud transaction Percentage--> ", percentage)

# SECTION B
cm = df[Numerical_features].corr()
print(df[cm])
fraud_corr = cm["isFraud"].abs().sort_values(ascending=False)
print("correlation values of all numerical features with isFraud", fraud_corr)

tcf = fraud_corr.head(4).index.tolist()
if "isFraud" in tcf:
    tcf.remove("isFraud")

sns.heatmap(cm)
plt.title("Correlation Matrix")
plt.show()
tcf = fraud_corr.drop("isFraud").head(3)
print("Top 3 Features correlated with Fraud", tcf)

# SECTION C
q1 = df["amount"].quantile(0.25)
q3 = df["amount"].quantile(0.75)
print(f"q1: {q1}, q3: {q3}")
IQR = q3 - q1
print(f"IQR: {IQR}")
lower_bound = q1 - 1.5 * IQR
upper_bound = q3 + 1.5 * IQR
print(f"Lower Bound: {lower_bound}\nUpper Bound: {upper_bound}")
# Pandas needs (cond1) | (cond2)
total_outliers = df[(df["amount"] > lower_bound) | (df["amount"] < upper_bound)]
print("Total outliers in Amount: ", len(total_outliers))
fraudulent_outliers = total_outliers[total_outliers["isFraud"] == 1]
print("Fradulent Outliers: ", len(fraudulent_outliers))
plt.figure(figsize=(15, 10))
sns.boxplot(y=df["amount"], color="lightblue")
plt.title("IQR Outlier Detection (Box Plot)", fontsize=16)

# SECTION D
print("\n--- Z score Outlier Detection for Top 5 features ---")
z_score_threshold = 3
top_5_features = fraud_corr.drop("isFraud").head(5).index
print(top_5_features)
for feature in top_5_features:
    mean_value = df[feature].mean()
    std_value = df[feature].std()

    df[f"{feature}_zscore"] = (df[feature] - mean_value) / std_value

    total_outliers_zscore = df[df[f"{feature}_zscore"].abs() > z_score_threshold]

    fradlent_outliers_zscore = total_outliers_zscore[
        total_outliers_zscore["isFraud"] == 1
    ]

    print(f"Feature: {feature}")
    print(f"Mean: {mean_value: .2f}")
    print(f"Standard Deviation: {std_value:.2f}")
    print(f" Z-score Threshold: {z_score_threshold}")
    print(f"Total Outliers (Z-score): {len(total_outliers_zscore)}")
    print(
        f"Fraudlent Outliers (Class=1) among Z-score outliers: {len(fradlent_outliers_zscore)}"
    )

plt.figure(figsize=(18, 7))


if "amount_zscore" not in df.columns:
    mean_amount = df["amount"].mean()
    std_amount = df["amount"].std()
    df["amount_zscore"] = (df["amount"] - mean_amount) / std_amount


z_score_threshold = 3
total_outliers_amount_zscore = df[df["amount_zscore"].abs() > z_score_threshold]
z_fraud = total_outliers_amount_zscore[total_outliers_amount_zscore["isFraud"] == 1]

plt.scatter(df["amount"], df["amount_zscore"], alpha=0.3, label="Normal")
plt.scatter(z_fraud["amount"], z_fraud["amount_zscore"], color="red", label="Fraud")

plt.xlabel("Amount")
plt.ylabel("Z-score")
plt.title("Amount vs Z-score")
plt.legend()
plt.show()
