# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 22:21:45 2024

@author: Mohamed Arafa
"""


DATAPATH=r"D:\DS Projects\H&M Product recommendation\Raw Data"
img_DATAPATH=r"D:\DS Projects\H&M Product recommendation\Raw Data\images"


import os
import cv2
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import LabelEncoder
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline, AutoTokenizer, AutoModel
#from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
import transformers
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, TFAutoModel
from transformers import BertTokenizer, TFBertModel
import tensorflow as tf
from bert.tokenization import FullTokenizer
import tensorflow as tf
import tensorflow_hub as hub
from bert.tokenization import FullTokenizer
from bert import BertModelLayer
import matplotlib.pyplot as plt

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


#Plot Customer Image     
def input_show(data):
    plt.title(f"Customer Purchase")
    plt.imshow(cv2.imread(data.iloc[0]["filepath"]))
    plt.show()
    
#plot results
def show_result(data):
    fig = plt.figure(figsize=(12, 8))

    input_show(data)

    for i in range(1, min(len(data), 13)):  # Start from the second element
        plt.subplot(3, 4, i)
        plt.imshow(cv2.imread(data.iloc[i]["filepath"]))

    plt.show()
    

#create Words embeddings database
def create_words_embedding_db(article_descriptions):
        word_embeddings_database = []
        for sentence in article_descriptions:
            tokens_database = tokenizer.tokenize(sentence)
            token_ids_database = tokenizer.convert_tokens_to_ids(tokens_database)
            input_ids_database = tokenizer.build_inputs_with_special_tokens(token_ids_database)
            input_ids_database = input_ids_database[:128] + [0] * max(0, 128 - len(input_ids_database))
            input_ids_tensor_database = tf.constant([input_ids_database])
            bert_output_database = bert_model(input_ids_tensor_database)
            word_embeddings_database.append(tf.reshape(bert_output_database.last_hidden_state, [bert_output_database.last_hidden_state.shape[1], bert_output_database.last_hidden_state.shape[2]]))
        return word_embeddings_database 
    
    
#create Word embedding

def create_word_embedding(input_text):
    tokens_input = tokenizer.tokenize(input_text)
    token_ids_input = tokenizer.convert_tokens_to_ids(tokens_input)
    input_ids_input = tokenizer.build_inputs_with_special_tokens(token_ids_input)
    input_ids_input = input_ids_input[:128] + [0] * max(0, 128 - len(input_ids_input))

    # Convert to TensorFlow tensor
    input_ids_tensor_input = tf.constant([input_ids_input])

    # Get BERT embeddings for the input sentence
    bert_output_input = bert_model(input_ids_tensor_input)
    word_embeddings_input = bert_output_input.last_hidden_state
    word_embeddings_input_reshaped = tf.reshape(word_embeddings_input, [word_embeddings_input.shape[1], word_embeddings_input.shape[2]])
    return word_embeddings_input_reshaped
    
#load model    
def load_model():
    bert_model = TFBertModel.from_pretrained("bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")    
    
    return(bert_model,tokenizer)

#find similar
def find_similar(database,sentence):
    similarities = [cosine_similarity(sentence, emb.numpy())[0, 0] for emb in database["tex_embeddings"]]
    most_similar_index = np.argsort(similarities)[::-1][:20]
    most_similar_sentences_id = [database.iloc[i] for i in most_similar_index]
    return pd.DataFrame(most_similar_sentences_id)

#recommend Items
def recommend_items(data):
    results = []  # Initialize an empty list to store results

    if isinstance(data, list):
        # Iterate over the list of sentences
        for input_text in data:
            word_embeddings = create_word_embedding(input_text)
            similar = find_similar(merged_df, word_embeddings)
            results.append(similar)  # Append the result to the list
    elif isinstance(data, str):
        # Process a single sentence
        word_embeddings = create_word_embedding(data)
        similar = find_similar(merged_df, word_embeddings)
        results.append(similar)  # Append the result to the list
    else:
        raise ValueError("Input data must be either a list of sentences or a single sentence.")

    # Visualize all results outside the loop
    for result in results:
        show_result(result)
    return results
        


articles_raw=pd.read_csv(os.path.join(DATAPATH,"01articles.csv"))
customers=pd.read_csv(os.path.join(DATAPATH,"customers_df_proccessed.csv"))
transactions_train_raw=pd.read_csv(os.path.join(DATAPATH,"03transactions_train.csv"))
transactions_train= transactions_train_raw.tail(100)
articles=articles_raw.tail(500)
img_df=get_img_path(img_DATAPATH)
img_df['article_id'] = img_df['article_id'].astype(str).str.lstrip('0')
img_df["article_id"]=img_df["article_id"].astype(int)
article_descriptions = articles['detail_desc'].fillna('').tolist()
max_length = max([len(description.split()) for description in article_descriptions])


model,tokenizer=load_model()
word_embeddingsdb=create_words_embedding_db(article_descriptions)

text_df = pd.DataFrame({"article_id":articles["article_id"],"tex_embeddings": word_embeddingsdb}) 
merged_df=pd.merge(text_df,img_df , on="article_id",how="left")
merged_df=pd.merge(articles,merged_df,on="article_id",how="left") 

results=recommend_items(article_descriptions[50:55])

















