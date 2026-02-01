import pandas as pd
import requests
from bs4 import BeautifulSoup

'''part-A'''

data="/Users/ayushparoha/Documents/Data_Science/env/college/SuperStoreUS-2015.xlsx"
df=pd.read_excel(data)
print(df.head())

print(df.info())
print(df.describe())

df.rename(columns={"Discount": "Max_Discount"}, inplace=True)
print(df)

df.groupby("Product Category")["Sales"].sum()

df.groupby("Product Category")["Sales"].sum().sort_values(ascending=False).head(3)

df["10_discount"]=df["Sales"]*0.1
print(df["10_discount"])

df.info()

total_sales=df["Sales"].sum()
print((df.groupby('Product Category')['Sales'].sum()/total_sales*100).idxmax())

print(df["Product Category"].value_counts()) #its is the frequency of that item

#for Part-B
df=pd.read_csv("/Users/ayushparoha/Documents/Data_Science/env/college/World Happiness Report.csv")
print(df.head())

df.columns
df.dtypes

df.sort_values("Happiness Score", ascending=False).head(10)[["Country", "Happiness Score"]]
df.groupby("Happiness Score")["Country"].sort_values("Happiness Score", ascending=False).head(10)

df.groupby("Region")["Happiness Score"].mean()

#Part-c Web-scraping
url="https://books.toscrape.com/"
res=requests.get(url)
print(res.status_code)
title=[]
prices=[]
try:
    soup=BeautifulSoup(res.text,"html.parser")
    data=soup.find_all("h3")
    title.extend([t.find("a")["title"].strip() for t in data])
    print(title)

    data1=soup.find_all("p", class_="price_color")
    prices.extend([p.text.strip() for p in data1])
    print(prices)

except requests.RequestException as e:
    print("Permission denied:", e)

data={
    "Book_Tittle": title,
    "Amount": prices
}

df=pd.DataFrame(data)
print(df.head())


