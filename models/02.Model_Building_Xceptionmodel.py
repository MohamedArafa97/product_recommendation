# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 21:58:43 2024

@author: Mohamed Arafa
"""
DATAPATH=r"D:\DS Projects\H&M Product recommendation\Raw Data"
img_DATAPATH=r"D:\DS Projects\H&M Product recommendation\Raw Data\images"

import os
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer
from keras.applications.xception import Xception,preprocess_input
import tensorflow as tf
from keras.preprocessing import image
from keras.layers import Input
from keras.backend import reshape
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split





#helper Functions 


def get_img_path(path):
    articles = {"article_id": [], "filepath": []}

    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            article_id = filename.split('.')[0]  
            imagepath = os.path.join(dirname, filename)

            articles["article_id"].append(article_id)
            articles["filepath"].append(imagepath)

    return pd.DataFrame(articles)


#load model
def model():
    model=Xception(weights='imagenet',include_top=False)
    for layer in model.layers:
        layer.trainable=False
        #model.summary()
    return model

#preprocess images
def preprocess_img(img_path):
    dsize = (225,225)
    new_image=cv2.imread(img_path)
    new_image=cv2.resize(new_image,dsize,interpolation=cv2.INTER_NEAREST)  
    new_image=np.expand_dims(new_image,axis=0)
    new_image=preprocess_input(new_image)
    return new_image


#Feature Embeddings
def feature_extraction(image_data,model):
    #Xmodel=model()
    img=preprocess_img(image_data)
    features=Xmodel.predict(img)
    features=np.array(features)
    features=features.flatten()
    return features


def feature_extraction_parallel(img_path):
    Xmodel = model()  # Load the model inside the function
    img = preprocess_img(img_path)
    features = Xmodel.predict(img)
    features = np.array(features)
    features = features.flatten()
    return features

with concurrent.futures.ProcessPoolExecutor() as executor:
    image_features = list(executor.map(feature_extraction_parallel, image_files_in_transactions["filepath"]))


    
    
    

#Plot Customer Image     
def input_show(data,result,i):
    plt.title(f"Customer {i+1}")
    index_result = result.index[0]
    filepath = data[data["article_id"] == index_result]["filepath"].iloc[0]
    plt.imshow(cv2.imread(filepath))
    plt.show()


#Plot Results
def show_result(data, result,custno):
    fig = plt.figure(figsize=(12, 8))
    input_show(data,result,custno)
    for i in range(len(result)):
        index_result = result.index[i]
        filepath = data[data["article_id"] == index_result]["filepath"].iloc[0]
        plt.subplot(3, 4, i + 1)
        
        plt.imshow(cv2.imread(filepath))
    
    plt.show()

#recommend items
def recommend_items(data,transactions):
    
    recommendations_list = []
    
    for i in range(len(transactions)):
        article_row=transactions.iloc[i]["article_id"]
        recommend_items=cosine_sim_df[article_row].sort_values(ascending=False)
        recommendations=recommend_items.nlargest(12)
        show_result(img_df, recommendations,i)
        recommendations_list.append({"customer_id": transactions.iloc[i]["customer_id"], "recommendations": np.array(recommendations.index)})
        recommendations_df = pd.DataFrame(recommendations_list)
        
    recommendations_df = pd.DataFrame(recommendations_list)
    return recommendations_df
    



articles=pd.read_csv(os.path.join(DATAPATH,"01articles.csv"))
customers=pd.read_csv(os.path.join(DATAPATH,"customers_df_proccessed.csv"))
transactions_train_raw=pd.read_csv(os.path.join(DATAPATH,"03transactions_train.csv"))
transactions_train = transactions_train_raw.tail(600)
unique_cut=transactions_train["customer_id"].unique()
img_df=get_img_path(img_DATAPATH)



# Remove leading zeros from 'article_id' column

img_df['article_id'] = img_df['article_id'].astype(str).str.lstrip('0')
img_df["article_id"]=img_df["article_id"].astype(int)



#get Images in Sample Only 

unique_articles=transactions_train["article_id"].unique()
image_files_in_transactions=img_df[img_df["article_id"].isin(unique_articles)]
    
    
Xmodel=model()    
image_features = [feature_extraction(img_path,Xmodel) for img_path in image_files_in_transactions["filepath"]]


image_df = pd.DataFrame({"article_id":image_files_in_transactions["article_id"],"image_features": image_features})

image_features

#user_item_matrix

merged_df = pd.merge(transactions_train, articles, on="article_id", how="inner")

user_item_matrix = merged_df.pivot_table(index="customer_id", columns="article_id", values="price", aggfunc="count", fill_value=0)


#train test split

train_set, holdout_set = train_test_split(transactions_train, test_size=20, random_state=42)

#article embeddings and cosine similarity

article_embeddings = image_df[image_df['article_id'].isin(user_item_matrix.columns)]
article_embeddings_matrix = np.array(list(article_embeddings['image_features']))

cosine_sim = cosine_similarity(article_embeddings_matrix, article_embeddings_matrix)
cosine_sim_df = pd.DataFrame(cosine_sim, index=article_embeddings['article_id'], columns=article_embeddings['article_id'])

rec_df=recommend_items(img_df,holdout_set)

