# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 17:11:49 2022

@author: asus
"""

# Data Scraping Process #

#conda install -c conda-forge textblob
# pip install snscrape
# pip install transformers
# conda install pytorch torchvision torchaudio cpuonly -c pytorch
# conda install -c conda-forge hvplot

# for Python 2: use print only as a function
from __future__ import print_function

# import and instantiate CountVectorizer (with the default parameters)
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from math import log, sqrt
import pandas as pd
import numpy as np
import re
%matplotlib inline
import string
import random
import networkx as nx
import nltk
import snscrape.modules.twitter as sntwitter
import hvplot.pandas
from transformers import pipeline

nltk.download('stopwords')
nltk.download('punkt') #punkt tokenizer model
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import FreqDist, bigrams
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

import numpy as np
from colorama import Fore, Style
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, f1_score, recall_score


# Data Collection #

sentiment_pipeline = pipeline(model='cardiffnlp/twitter-roberta-base-sentiment')

def tweets(n_tweets, search_term, start_date, end_date):
    """
    get a dataframe of tweets by search term
    
    ref: https://betterprogramming.pub/how-to-scrape-tweets-with-snscrape-90124ed006af
    """
    # Creating list to append tweet data to
    tweets_list2 = []

    # Using TwitterSearchScraper to scrape data and append tweets to list
    for i,tweet in enumerate(sntwitter.TwitterSearchScraper(f'{search_term} since:{start_date} until:{end_date}').get_items()):
        if i>n_tweets:
            break
        tweets_list2.append([tweet.date, tweet.id, tweet.content])

    # Creating a dataframe from the tweets list above
    tweets_df2 = pd.DataFrame(tweets_list2, columns=['Datetime', 'Tweet Id', 'Text'])
    return tweets_df2

df_amber = tweets(10000, 'amber heard', '2022-05-18', '2022-05-20')

# Place into Data Frame
df_amber.to_csv("C:/Users/ASUS/Downloads/SMA_Amber.csv")

     




















