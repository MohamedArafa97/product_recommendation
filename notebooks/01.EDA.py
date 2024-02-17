# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 15:59:26 2024

@author: Mohamed Arafa
"""

DATAPATH=r"D:\DS Projects\H&M Product recommendation\Raw Data"
IMGPATH=r"D:\DS Projects\H&M Product recommendation\Visuals"


import pandas as pd
import numpy as np 
import pickle
import logging 
import os 

import matplotlib.pyplot as plt
import plotly.express as px
import squarify    # pip install squarify (algorithm for treemap)
from scipy.cluster.hierarchy import dendrogram
import seaborn as sns 
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.figure_factory as ff
from matplotlib.cm import viridis
from matplotlib.dates import DateFormatter



articles_raw=pd.read_csv(os.path.join(DATAPATH,"01articles.csv"))
articles_df=articles_raw.copy()
articles_df.describe()
articles_df.columns
articles_df.dtypes


#unique_values

uniques=pd.DataFrame()
for i in articles_df.columns:
    unique_values_count = articles_df[i].nunique()
    uniques["unique_"+i]=[unique_values_count]

print(uniques.columns)

#article_id=105542
#priducts=47224
#product_type=132
#product_group=19
#physical_appearance=30
#color_group=50
#perceived_colour_value_name=8
#perceived_colour_master_name=20
#department_name=250


#columns unique values exploration

prodcuts=articles_df["prod_name"]
product_type=articles_df["product_type_name"].unique()
product_groups=articles_df["product_group_name"].unique()
product_apperances=articles_df["graphical_appearance_name"].unique()
product_colors=articles_df["colour_group_name"].unique()
product_colors_perceived=articles_df["perceived_colour_value_name"].unique()
product_Departments=articles_df["department_name"].unique()

#missing values 

articles_df.isna().sum()/len(articles_df)*100

#about 0.4% of articles has no description 
#no other missing values



#volume of each product 

temp = articles_df.groupby(["product_group_name"])["product_type_no"].nunique()
print(temp)
df = pd.DataFrame({'Product Group': temp.index,'Product Types': temp.values})
df = df.sort_values(['Product Types'], ascending=False)
print(df)

#bar chart for volume of each product code

plt.figure(figsize=(12, 6))
colors = viridis(df['Product Types'] / df['Product Types'].sum()) 
plt.bar(df["Product Group"], df["Product Types"], color= colors)
plt.xlabel('Product Code')
plt.ylabel('Volume')
plt.xticks(rotation=90) 
plt.title('Volume of Each Product Code')
plt.savefig((os.path.join(IMGPATH,'volume_product_code_barchart.png')),dpi=300, bbox_inches='tight', format='png')

#pie chart 

plt.figure(figsize=(13, 8))
plt.title('Volume of Each Product Code (Pie Chart)')
plt.pie( df["Product Types"], labels=df['Product Group'], colors=colors)
plt.savefig((os.path.join(IMGPATH,'volume_product_code_piechart.png')),dpi=300, bbox_inches='tight', format='png')


#volum of each index group name
 
temp = articles_df.groupby(["index_group_name"])["article_id"].nunique()
df = pd.DataFrame({'Index Group Name': temp.index,'Articles': temp.values})
df = df.sort_values(['Articles'], ascending=False)
print(df)

plt.figure(figsize=(12, 6))
print(df['Articles'].sum())
colors = viridis(df['Articles'] / df['Articles'].sum())  # Normalize values to [0, 1]

bars = plt.bar(df["Index Group Name"], df["Articles"], color=colors)
for bar, value in zip(bars, df["Articles"]):
    height = bar.get_height()
    percentage = (value / df["Articles"].sum()) * 100  # Calculate percentage
    plt.text(bar.get_x() + bar.get_width() / 2, height,f'{int(percentage)}%', ha='center', va='bottom')

plt.xlabel('Index Group')
plt.ylabel('Volume')
plt.xticks(rotation=90) 
plt.title('Volume of Each Index Group')
plt.savefig((os.path.join(IMGPATH,'volume_index_group_barchart.png')),dpi=300, bbox_inches='tight', format='png')

#volume of each product type 

temp = articles_df.groupby(["product_type_name"])["article_id"].nunique()
df = pd.DataFrame({'Product Type': temp.index,'Articles': temp.values})
total_types = len(df['Product Type'].unique())
df = df.sort_values(['Articles'], ascending=False)[0:20]

plt.figure(figsize=(25, 11))
sns.barplot(data=df,x='Product Type', y='Articles',palette="magma")
plt.xlabel('Product Type',fontsize=30)
plt.ylabel('Volume',fontsize=30)
plt.xticks(rotation=45,fontsize=16)
plt.yticks(fontsize=16)
plt.title('Volume of Each Product type',fontsize=30)
plt.savefig((os.path.join(IMGPATH,'volume_prodcut_type_barchart.png')),dpi=300, bbox_inches='tight', format='png')


#customer Dataset

customers_raw=pd.read_csv(os.path.join(DATAPATH,"02customers.csv"))
customers_df=customers_raw.copy()
customers_df.describe()
customers_df.columns
customers_df.dtypes

customer_uniques=pd.DataFrame()
for i in customers_df.columns:
    unique_values_count = customers_df[i].nunique()
    customer_uniques["unique_"+i]=[unique_values_count]
    
print(customer_uniques.columns)


#converting NONE to None
customers_df["fashion_news_frequency"] = customers_df["fashion_news_frequency"].replace('None', "NONE")


#customers age distrubution

bins=[15, 20, 30, 40,50, 60, 70, float('Inf')]
labels=['16-20', '20-30','30-40','40-50','50-60','60-70' , '70+']
customers_df['age_group'] = pd.cut(customers_df['age'], bins=bins, labels=labels, right=False)

     
temp=customers_df.groupby(["age_group"])["customer_id"].count()
df=pd.DataFrame({"Age Group": temp.index , "Number of Customer" : temp.values})
df=df.sort_values(["Number of Customer"],ascending=False)
print(df)

      
plt.figure(figsize=(20, 10))
ax =sns.barplot(data=df,x='Age Group', y='Number of Customer',palette="magma")

# Annotate each bar with its percentage value
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2., height + 0.05, f'{int(height/df["Number of Customer"].sum()*100)}%', ha="center", va="bottom", fontsize=12)
    
    
plt.xlabel('Age Group',fontsize=16)
plt.ylabel('Number of Customer',fontsize=16)
plt.xticks(rotation=90,fontsize=16)
plt.yticks(fontsize=16)
plt.title('Number of Customer per age Group',fontsize=16)
plt.savefig((os.path.join(IMGPATH,'age_group_barchart.png')),dpi=300, bbox_inches='tight', format='png')


    
#Fashion news frequency 


temp=customers_df.groupby(["fashion_news_frequency"])["customer_id"].count()
df=pd.DataFrame({"FN Frequency": temp.index , "Number of Customer" : temp.values})
df=df.sort_values(["Number of Customer"],ascending=False)
print(df)

plt.figure(figsize=(20, 10))
sns.barplot(data=df,x='FN Frequency', y='Number of Customer',palette="magma")
plt.xlabel('FN Frequency',fontsize=16)
plt.ylabel('Number of Customer',fontsize=16)
plt.xticks(rotation=90,fontsize=16)
plt.title('Number of Customer per FN Frequency',fontsize=16)
plt.savefig((os.path.join(IMGPATH,'FN_freq_barchart.png')),dpi=300, bbox_inches='tight', format='png')


#transcation Data 

transactions_raw=pd.read_csv(os.path.join(DATAPATH,"03transactions_train.csv"))
train_df=transactions_raw.copy()
train_df.describe()
train_df.columns
train_df.dtypes

transactions_uniques=pd.DataFrame()
for i in train_df.columns:
    unique_values_count = train_df[i].nunique()
    transactions_uniques["unique_"+i]=[unique_values_count]
    
    
train_df.isna().sum()
#no missing values in the transactions data


#unsold articles 

article_sold= train_df["article_id"]
article_sold.drop_duplicates(inplace=True)
total_articles=articles_raw["article_id"]
total_articles.drop_duplicates(inplace=True)
unsold_article_ids = total_articles[~total_articles.isin(article_sold)]
unsold_articles=articles_raw[articles_raw["article_id"].isin(unsold_article_ids)]
temp=unsold_articles.groupby("index_group_name")["article_id"].count()
print(temp)


df=pd.DataFrame({"Group Name": temp.index , "Number of Unsold Articles" : temp.values})
df=df.sort_values(["Number of Unsold Articles"],ascending=False)
print(df)

      
plt.figure(figsize=(20, 10))
sns.barplot(data=df,x='Group Name', y='Number of Unsold Articles',palette="magma")
plt.xlabel('Group Name',fontsize=20)
plt.ylabel('Number of Unsold Articles',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.title('Number of Unsold Articles per Group Name',fontsize=20)
plt.savefig((os.path.join(IMGPATH,'unsold_articles_barchart.png')),dpi=300, bbox_inches='tight', format='png')



#average spending in one day per customer 

total_spending_per_customer =train_df.groupby(["t_dat","customer_id"])["price"].sum().reset_index()
print(total_spending_per_customer)
average_spending_per_day=total_spending_per_customer.groupby("t_dat")["price"].mean().reset_index()
print(average_spending_per_day)
total_spending_per_day=total_spending_per_customer.groupby("t_dat")["price"].sum().reset_index()
print(total_spending_per_day)
total_spending_per_day.mean()
#average Transactions per day is 1205 



#transactions distrubution over time

total_spending_per_day["t_dat"] = pd.to_datetime(total_spending_per_day["t_dat"]).dt.floor('D')
total_spending_per_day.set_index("t_dat",inplace=True)
df_resampled = total_spending_per_day[["price"]].resample('MS').sum()
df_resampled.reset_index(inplace=True)
df_resampled["t_dat"]=df_resampled["t_dat"].dt.strftime('%Y-%m')
df_resampled.set_index("t_dat",inplace=True)

plt.figure(figsize=(20, 10))
ax=sns.lineplot(data=df_resampled,x=df_resampled.index ,y="price")
ax.set_title('Total Spending per Month (Line Plot)',fontsize=20)
ax.set_xlabel('Date',fontsize=20)
ax.set_ylabel('Total Spending',fontsize=20)
ax.set_xticklabels(df_resampled.index,rotation=45,fontsize=20)
plt.savefig((os.path.join(IMGPATH,'transactions_permonth_lineplot.png')),dpi=300, bbox_inches='tight', format='png')

#Seosnality is present in the plot 
#Transactions tends to increase at the beginning of the year 
#Transactions tends to decrease at mid year (6-7-8)


#Top 100 for Generating Earning 

df_prices=train_df[["price","article_id"]].groupby("article_id").sum().sort_values("price",ascending=False)

print("Number of different sold articles:",len(df_prices["price"]))
print("Total Earnings:",df_prices["price"].sum())


for i in [10,50,100,200,300,400,1000]:
    print("The Top %s of products the generates most earning , account for the %.2f %% of the total earning" % (i,df_prices["price"].iloc[:i].sum() / df_prices["price"].iloc[:].sum() * 100))

top_100_prices=df_prices.iloc[:100]

top_100_prices_details=top_100_prices.merge(articles_raw,on=["article_id"],how="left")


#lowest earning Items 

worst_100_prices=df_prices.iloc[-100:]

worst_100_prices_details=worst_100_prices.merge(articles_raw,on=["article_id"],how="left")


#quantity sold per article 

df_sold_qty=train_df["article_id"].value_counts().reset_index()
df_sold_qty=df_sold_qty.rename(columns={"article_id" : "sold_quantity","index" : "article_id"})
df_sold_qty.describe()

#104547 different articles sold 
#average qtys sold is 305
#25% of sold products, have been sold 14 or less times
#50% were sold 65 or less times
#75% were sold 286 or less times,
#The most sold item have been sold 50287 times.
#There are items which have been sold only once


plt.figure(figsize=(12,4))
plt.title("Sold Quantity KDE plot")
sns.kdeplot(df_sold_qty["sold_quantity"])
plt.xlabel("Sold Quantity")
plt.show()

#Sold quantity are heavily right skewed

df_sold_qty["sold_quantity"].quantile([0.90,0.95,0.99,0.999])

#90% of the articles have been sold 793 or less times
#95% of the articles have been sold 1318 or less times
#99% of the articles have been sold 3185 or less times
#a small fraction is sold more than 10000 times which is less than 1%




#top 100 items sold 

top_100_df=df_sold_qty.sort_values("sold_quantity",ascending=False).iloc[:100]
top_100_details=top_100_df.merge(articles_raw,on=["article_id"],how="left")

#top 100 by index group 

top_100_by_index_group=top_100_details.groupby("index_group_name")["sold_quantity"].sum().reset_index().sort_values("sold_quantity",ascending=False)
top_100_by_index_group["qty%"]=(top_100_by_index_group["sold_quantity"]/top_100_by_index_group["sold_quantity"].sum())*100

#top 100 by index 

top_100_by_index_name=top_100_details.groupby("index_name")["sold_quantity"].sum().reset_index().sort_values("sold_quantity",ascending=False)
top_100_by_index_name["qty%"]=(top_100_by_index_name["sold_quantity"]/top_100_by_index_name["sold_quantity"].sum())*100

#top 100 by Product type 

top_100_by_product_type_name=top_100_details.groupby("product_type_name")["sold_quantity"].sum().reset_index().sort_values("sold_quantity",ascending=False)
top_100_by_product_type_name["qty%"]=(top_100_by_product_type_name["sold_quantity"]/top_100_by_product_type_name["sold_quantity"].sum())*100


#top 100 by Product Group 

top_100_by_product_group_name=top_100_details.groupby("product_group_name")["sold_quantity"].sum().reset_index().sort_values("sold_quantity",ascending=False)
top_100_by_product_group_name["qty%"]=(top_100_by_product_group_name["sold_quantity"]/top_100_by_product_group_name["sold_quantity"].sum())*100

#top 100 by Color Master Group

top_100_by_colour_master=top_100_details.groupby("perceived_colour_master_name")["sold_quantity"].sum().reset_index().sort_values("sold_quantity",ascending=False)
top_100_by_colour_master["qty%"]=(top_100_by_colour_master["sold_quantity"]/top_100_by_colour_master["sold_quantity"].sum())*100



#top 40 most sold items 

plt.figure(figsize=(10,8))
plt.title("TOP 40 most sold products", fontsize=33, fontweight="bold")
no=40
g = sns.barplot(y="prod_name", x="sold_qty(%)", data=top_100_details.iloc[:no].groupby("prod_name")["sold_quantity"].sum() \
            .transform(lambda x: (x / x.sum() * 100)).rename('sold_qty(%)').reset_index().sort_values(by="sold_qty(%)", ascending=False), \
            palette="mako", ci=False)
for container in g.containers:
    g.bar_label(container, padding = 5, fmt='%.f%%', fontsize=12)
plt.xlabel("Sold Quantity (%)", size=25, fontweight="bold")
plt.ylabel("")
plt.grid(axis="x",color = 'grey', linestyle = '--', linewidth = 1.5)
plt.savefig((os.path.join(IMGPATH,'top_sold_products.png')),dpi=300, bbox_inches='tight', format='png')


#helper Function

def make_barplot(data,title,filename):
        
    plt.figure(figsize=(10,8))
    plt.title(title, fontsize=33, fontweight="bold")
    
    x=data.iloc[:,2]
    y=data.iloc[:,0]
    
    g = sns.barplot(y=y, x=x, data=data,palette="mako", ci=False)
    for container in g.containers:
        g.bar_label(container, padding = 5, fmt='%.f%%', fontsize=12)
    plt.xlabel("Sold Quantity (%)", size=25, fontweight="bold")
    plt.ylabel("")
    plt.grid(axis="x",color = 'grey', linestyle = '--', linewidth = 1.5)
    plt.savefig((os.path.join(IMGPATH, filename)),dpi=300, bbox_inches='tight', format='png')
    plt.show()
    
    
  
make_barplot(top_100_by_colour_master,"TOP 100 most sold products","top_sold_products_bycolor.png")
make_barplot(top_100_by_product_group_name,"TOP 100 most sold products","top_sold_products_bygroup.png")
make_barplot(top_100_by_product_type_name,"TOP 100 most sold products","top_sold_products_bytype.png")
make_barplot(top_100_by_index_name,"TOP 100 most sold products","top_sold_products_by_index_name.png")
make_barplot(top_100_by_index_group,"TOP 100 most sold products","top_sold_products_by_index_name.png")

#first 4 most sold items are responsible of 42% of sales(40 products) and 22% os sales(100 products)
#most sold items 
#68% of most 100 sold items are ladiesware 
df_sold_qty.columns

#articles sold only once 
df_sold_qty[df_sold_qty["sold_quantity"] == 1 ] 

worst_items_df=df_sold_qty.tail(4491)
worst_items_details=worst_items_df.merge(articles_raw,on=["article_id"],how="left")

#top 100 by index group 

worst_items_index_group=worst_items_details.groupby("index_group_name")["sold_quantity"].sum().reset_index().sort_values("sold_quantity",ascending=False)
worst_items_index_group["qty%"]=(worst_items_index_group["sold_quantity"]/worst_items_index_group["sold_quantity"].sum())*100

#top 100 by index 

worst_items_index_name=worst_items_details.groupby("index_name")["sold_quantity"].sum().reset_index().sort_values("sold_quantity",ascending=False)
worst_items_index_name["qty%"]=(worst_items_index_name["sold_quantity"]/worst_items_index_name["sold_quantity"].sum())*100

#top 100 by Product type 

worst_items_product_type_name=worst_items_details.groupby("product_type_name")["sold_quantity"].sum().reset_index().sort_values("sold_quantity",ascending=False)
worst_items_product_type_name_30=worst_items_product_type_name[:30]
worst_items_product_type_name_30["qty%"]=(worst_items_product_type_name_30["sold_quantity"]/worst_items_product_type_name["sold_quantity"].sum())*100


#top 100 by Product Group 

worst_items_product_group_name=worst_items_details.groupby("product_group_name")["sold_quantity"].sum().reset_index().sort_values("sold_quantity",ascending=False)
worst_items_product_group_name["qty%"]=(worst_items_product_group_name["sold_quantity"]/worst_items_product_group_name["sold_quantity"].sum())*100

#top 100 by Color Master Group

worst_items_colour_master=worst_items_details.groupby("perceived_colour_master_name")["sold_quantity"].sum().reset_index().sort_values("sold_quantity",ascending=False)
worst_items_colour_master["qty%"]=(worst_items_colour_master["sold_quantity"]/worst_items_colour_master["sold_quantity"].sum())*100


make_barplot(worst_items_colour_master,"Items sold Once","sold_once_product_bycolor.png")
make_barplot(worst_items_product_group_name,"Items sold Once","sold_once_product_bygroup.png")
make_barplot(worst_items_product_type_name_30,"30 Items sold Once","sold_once_product_bytype.png")
make_barplot(worst_items_index_name,"Items sold Once","sold_once_product_index_name.png")
make_barplot(worst_items_index_group,"Items sold Once","sold_once_product_index_group.png")

#most items that are sold once are from Children sections 



#Customer Analysis

df_cust_prices=train_df[["customer_id","price"]].groupby("customer_id").sum()
df_cust_qty=train_df[["customer_id","article_id"]].groupby("customer_id").count()
customers_df=pd.read_csv(os.path.join(DATAPATH,"customers_df_proccessed.csv"))
customers_df.drop("Unnamed: 0",axis=1,inplace=True)
cust_qty_price = pd.merge(df_cust_prices, df_cust_qty, on='customer_id', how='inner')
cust_details = pd.merge(cust_qty_price, customers_df.drop("postal_code", axis=1), on='customer_id', how='inner')
cust_details.article_id.describe()

plt.figure(figsize=(10,4))
plt.title("Distribution of purchased quantity by customer", fontweight="bold", size=20)
sns.kdeplot(cust_details["article_id"])
plt.xlabel("purchased quantity",fontweight="bold", size=20)
plt.ylabel("Count",fontweight="bold", size=20)
plt.show()

#Purchased Qtys by age group

temp=cust_details.groupby("age_group")["article_id"].sum().reset_index()
temp["article_id"].sum()


plt.figure(figsize=(10,8))
plt.title("Purchased Qty by age group", fontsize=33, fontweight="bold")
g = sns.barplot(y=temp["article_id"]/temp["article_id"].sum()*100, x="age_group", data=temp,palette="mako", ci=False)
for container in g.containers:
    g.bar_label(container, padding = 5, fmt='%.f%%', fontsize=12)
plt.xlabel("Sold Quantity (%)", size=25, fontweight="bold")
plt.ylabel("")
plt.grid(axis="x",color = 'grey', linestyle = '--', linewidth = 1.5)
plt.savefig((os.path.join(IMGPATH, "quantities_by_age_group.png")),dpi=300, bbox_inches='tight', format='png')


#Purchased Qtys by Fashion news 

fashion_freq_qty=cust_details.groupby("fashion_news_frequency")["article_id"].sum().reset_index()

plt.figure(figsize=(10,8))
plt.title("Purchased Qty by FN frquency", fontsize=33, fontweight="bold")
g = sns.barplot(y=fashion_freq_qty["article_id"]/fashion_freq_qty["article_id"].sum()*100, x="fashion_news_frequency", data=fashion_freq_qty,palette="mako", ci=False)
for container in g.containers:
    g.bar_label(container, padding = 5, fmt='%.1f', fontsize=12)
plt.xlabel("fashion news frequency", size=25, fontweight="bold")
plt.ylabel("Purchased Qtys%", size=25, fontweight="bold")
plt.grid(axis="x",color = 'grey', linestyle = '--', linewidth = 1.5)
plt.savefig((os.path.join(IMGPATH, "quantities_by_FN_freq.png")),dpi=300, bbox_inches='tight', format='png')