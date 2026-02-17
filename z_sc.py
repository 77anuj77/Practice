#z-score

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("/Users/ayushparoha/Documents/Data_Science/env/college/ass4/creditcard.csv")
df.isnull().sum()
df.shape

#-q "path of unzip file"

#corellation matrix; .abs is the absolute value acta as a modulus function
#class corellation
cm=df.corr()
cc=cm["Class"].abs().sort_values(ascending=False)

print("Absolute correlation with 'class' in decending:\n ",cc)

sns.heatmap(cm)
plt.title('Correlation Matrix')
plt.show()

#for finding top 5 correlated features 
tcf=cc.head(6).index.tolist() #the index will in the form of list(like a row)
if 'Class' in tcf:
    tcf.remove('Class')

#top 5 features =t5f
t5f=tcf[:5]
print(t5f)


print("\n------------- IQR Outlier Detection for Top 5 Features -------------")

for feature in t5f:
    
    q1 = df[feature].quantile(0.25)
    q3 = df[feature].quantile(0.75)
    
    IQR = q3 - q1
    
    lower_bound = q1 - 1.5 * IQR
    upper_bound = q3 + 1.5 * IQR
    
    # Total outliers
    to = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]
    
    # Fraudulent outliers among total outliers
    fo = to[to["Class"] == 1]
    
    print(f"\nFeature: {feature}")
    print(f"Q1: {q1:.2f}")
    print(f"Q3: {q3:.2f}")
    print(f"IQR: {IQR:.2f}")
    print(f"Lower Bound: {lower_bound:.2f}")
    print(f"Upper Bound: {upper_bound:.2f}")
    print(f"Total Outliers: {len(to)}")
    print(f"Fraudulent Outliers (Class 1): {len(fo)}")



# Z-score Outlier Detection
print("\n---------- Z-Score Outlier Detection ----------")

#z-scores threshold
z = 3   # threshold

for feature in t5f:
    
    mean = df[feature].mean()
    std = df[feature].std()

    # Calculate Z-score
    df[f"{feature}_zscore"] = (df[feature] - mean) / std

    # Identify total outliers
    toz = df[df[f"{feature}_zscore"].abs() > z]

    # Identify fraudulent outliers
    foz = toz[toz["Class"] == 1]

    print(f"\nFeature: {feature}")
    print(f"Mean: {mean:.2f}")
    print(f"Standard Deviation: {std:.2f}")
    print(f"Z-score Threshold: {z}")
    print(f"Total Outliers: {len(toz)}")
    print(f"Fraudulent Outliers (Class 1): {len(foz)}")

    # Remove temporary z-score column
    df.drop(columns=[f"{feature}_zscore"], inplace=True)


#//////////////////////////////////////////////////////////////////////////////////
# Combine top 5 features + "Amount" for IQR visualization
feature_for_iqr_plot = t5f + ['Amount']

plt.figure(figsize=(15, 10))
plt.suptitle("IQR Outlier Detection (Box Plot)", fontsize=16)

#enumereate ->> execute values one-by-one
for i, feature in enumerate(feature_for_iqr_plot):
    plt.subplot(2, 3, i + 1)

    sns.boxplot(y=df[feature], color="lightblue")

    # Overlay fraudulent transactions
    fraudulent_values = df[df['Class'] == 1][feature]

    #zorder->>> giving a ordering sequence top priority and 1 as least priority
    plt.scatter(x=np.zeros(len(fraudulent_values)),y=fraudulent_values,color='red',s=20, label="Fraudulent (Class 1)" if i == 0 else "",zorder=5)

    plt.title(f"Box Plot of {feature}")
    plt.ylabel(feature)
    
    #hide the x-ticks of x axis
    plt.xticks([])
    
    if i == 0:
        plt.legend()

#tight_layout for proper spacing between graphs
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
#/////////////////////////////////////////////////////////////////////////////////////

# Combine top 5 features + "Amount" for IQR visualization
feature_for_iqr_plot = t5f + ['Amount']

plt.figure(figsize=(15, 10))
plt.suptitle("IQR Outlier Detection (Box Plot)", fontsize=16)

rows = 2
cols = 3

for i, feature in enumerate(feature_for_iqr_plot):
    plt.subplot(rows, cols, i + 1)

    sns.boxplot(y=df[feature], color="lightblue")

    # Overlay fraudulent transactions
    fraudulent_values = df[df['Class'] == 1][feature]

    plt.scatter(
        x=np.zeros(len(fraudulent_values)),
        y=fraudulent_values,
        color='red',
        s=20,
        label="Fraudulent (Class 1)" if i == 0 else "",
        zorder=5
    )

    plt.title(f"Box Plot of {feature}")
    plt.ylabel(feature)
    plt.xticks([])

    if i == 0:
        plt.legend()

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()




#for scatter plot