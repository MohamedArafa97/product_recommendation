# Product Recommendation

## Overview

This project focuses on building a recommendation system for a clothing and fashion store based on customer's previous purchase history. Two different models have been implemented: Xception model for image data and BERT model for textual descriptions.

## Data Files

- **articles.csv**: Information about articles in the store.
- **customers.csv**: Information about customers.
- **transactions_train.csv**: Information about customer transactions.
- **Images**: Images for every article (product).

## Code Files

- **02.Model_Building_Xceptionmodel.ipynb**: Implementation of the recommendation system using the Xception model.
- **03.Model_Building_Bertmodel.ipynb**: Implementation of the recommendation system using the BERT model.
- **01.EDA.ipynb**: Code for exploratory data analysis.

## Figures

- Various images generated during EDA for better visualization and understanding.

## Exploratory Data Analysis (EDA)

### Articles Dataset

**Summary Statistics:**
- Number of unique articles: 105,542
- Number of unique products: 47,224
- Number of unique product types: 132
- Number of unique product groups: 19
- Number of unique physical appearances: 30
- ...

**Missing Values:**
- About 0.4% of articles have no description, but there are no other missing values.

**Volume Analysis:**
- Products categorized into groups visualized using bar and pie charts.

### Customer Dataset

- Age distribution of customers.
- Fashion news frequency distribution among customers.

### Transaction Dataset

- Visualization of unsold articles.
- Average spending per day per customer.
- ...

**Top Performing and Least Performing Products:**
- Identification of top 100 products generating the most earnings.
- Identification of worst-performing products (unsold and sold once).

**Customer Analysis:**
- Distribution of purchased quantity by customers.
- Analysis of purchased quantities based on customer age groups and fashion news frequency.

## Xception Model Recommendation System

This section focuses on implementing a recommendation system using the Xception model. Recommendations are based on customer preferences inferred from historical transactions.

### Data Loading and Preprocessing

1. Load articles, customer data, and transaction data.
2. Retrieve image paths for articles.

### Feature Extraction

1. Extract feature embeddings using the Xception model for images associated with articles in the transaction data.
2. Create a dataframe with article IDs and their corresponding image features.

### Article Embeddings and Cosine Similarity

1. Extract article embeddings for articles in the dataset.
2. Calculate cosine similarity between article embeddings.

### Recommendation System

Generate recommendations for each customer in the test set and visualize the input images along with the recommended items.

This Xception model-based recommendation system aims to enhance customer satisfaction and increase sales by providing personalized product suggestions based on their preferences. The combination of image embeddings and cosine similarity contributes to the effectiveness of the recommendation engine.

## BERT Model Recommendation System

This part involves implementing a recommendation system using a BERT-based model. Recommendations are based on the textual descriptions of articles.

### Data Loading and Preprocessing

1. Load articles, customer data, and transaction data.
2. Extract unique customer IDs from the transactions.

### Word Embeddings

1. Load the BERT model and tokenizer.
2. Create a database of word embeddings for article descriptions using BERT.

### Recommendation System

1. Merge BERT word embeddings with image data and article information.
2. Generate recommendations for a list of article descriptions.

This BERT model-based recommendation system leverages word embeddings to capture the semantic meaning of article descriptions. The recommendation engine aims to enhance customer satisfaction by providing personalized product suggestions based on textual information. The combination of image and text-based recommendation systems can offer a comprehensive and tailored shopping experience for customers.



Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
