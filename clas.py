#IQR inter quartile range (lower bound and upper bound, median,q1 &q3)
#z-scores(mean and deviation for finding the outlier)
#visulise anamalies- boxplot(IQR) and scaterplot(z-score)
#class-primary data that is important ;feaures-that are directly affecting the data of class if changed

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#-q "path of unzip file"

df=pd.read_csv("/Users/ayushparoha/Documents/Data_Science/env/ass4/creditcard.csv")
df.isnull().sum()
df.shape

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