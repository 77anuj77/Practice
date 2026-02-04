import pandas as pd
import numpy as np

url="data.csv"
data=pd.read_csv(url,encoding="latin1")
print(data)

print(data.shape)

print(data.info())

#check the count of the duplicate rows
data.duplicated().sum()

#removing duplicates row
data.drop_duplicates(inplace=True)

print(data.shape)

#verifying all duplicates are removed
print(data.duplicated().sum())

#there are to different null value Nan and NaT
import pandas as pd

# Assuming your DataFrame is named 'data'
data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"], errors="coerce")#this will put nat in invalid data and not permanenty puting its value

#fixing quantity and unit price eith -ve numeric values
data["Quantity"]=pd.to_numeric(data["Quantity"], errors="coerce")
data["UnitPrice"]=pd.to_numeric(data["UnitPrice"], errors="coerce")

#removing 0 values
data=data[data["Quantity"]>0]
data=data[data["UnitPrice"]>0]

#
data.dropna(subset=["CustomerID"])

data["Description"]=data["Description"].fillna("Unknown Product")

print(data["Description"])

print(data.info())
print(data.isnull().sum())