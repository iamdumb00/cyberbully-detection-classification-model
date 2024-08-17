# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 19:22:58 2022

@author: asus
"""
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


# Data Cleaning #

tweets=pd.read_csv("C:/Users/ASUS/Downloads/SMA_Amber.csv")

# Lemmatize function 
def lemmatize_sentence(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmatized_sentence = []
    for word, tag in pos_tag(tokens):
        if tag.startswith('NN'):    #Noun
            pos = 'n'
        elif tag.startswith('VB'):  #Verb
            pos = 'v'
        else:
            pos = 'a'               #Adjective
        lemmatized_sentence.append(lemmatizer.lemmatize(word, pos))
    return lemmatized_sentence

def remove_noise(tweets_tokens, stop_words):
    cleaned_tokens = []
    for token in tweets_tokens:
        token = re.sub('http[s]','', token)
        token = re.sub('//t.co/[A-Za-z0-9]+','', token)
        token = re.sub('(@[A-Za-z0-9_]+)','', token)
        token = re.sub('[0-9]','', token)
        if (len(token) > 3) and (token not in string.punctuation) and (token.lower() not in stop_words):
            cleaned_tokens.append(token.lower())
        return cleaned_tokens
    
# Common but unimportant words
stop_words = stopwords.words('english')
stop_words.extend(['amber', 'Amber' , 'Heard', 'AmberHeard',
                   'johnny', 'Johnny', 'Depp', 'JohnnyDepp',
                   'julia', 'ellen', 'camille', 'estoy', 'https'])

tweets_token=tweets['Text'].apply(word_tokenize).tolist()

cleaned_tokens = []
for tokens in tweets_token:
    rm_noise = remove_noise(tokens, stop_words)
    lemma_tokens = lemmatize_sentence(rm_noise)
    cleaned_tokens.append(lemma_tokens)

def get_all_words(cleaned_tokens_list):
    tokens = []
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token

tokens_flat = get_all_words(cleaned_tokens)

freq_dist = FreqDist(tokens_flat)


# Print most common words
print(freq_dist.most_common(20))

# --------- finished data cleaning process -------- #

# introduce N-GRAM MODEL, machine learning 
# unigram = 1 word, bigram = 2 word, trigram = 3 word, quadgrams = 4 word
# ^ to show the relationship between the words, so we can identify it
# the more words tgt, the relationship between words would be more clearer and meaningful

# -------------- BIGRAM --------------- #
# what words often show up together 
bigram_list = [list(bigrams(Text)) for Text in cleaned_tokens]

bigrams_flat = get_all_words(bigram_list)

freq_dist_bigrams = FreqDist(bigrams_flat)

print(freq_dist_bigrams.most_common(10))

# visualize the relationship between words thru networkx $ matplotlib
# create nwtwork graph
# top 50 most commmonly used 
network_token_df = pd.DataFrame(freq_dist_bigrams.most_common(50), columns=['token', 'count'])

#convert bigrams to a single dictionary
bigrams_d = network_token_df.set_index('token').T.to_dict('records')

# Label response type from tweets
text_blob=[] 
for tweet in tweets['Text'].tolist():
    analysis = TextBlob(tweet)
    if analysis.sentiment.polarity == 0:
        sentiment = "Neutral"
    elif analysis.sentiment.polarity > 0:
        sentiment = "Positive"       
    elif analysis.sentiment.polarity < 0:
        sentiment = "Negative"       
    text_blob.append(sentiment)
    
tweets['Sentiment'] = text_blob

# Drop neutral tweets (only positive and negative)
labelled_tweets = tweets[['Text', 'Sentiment']]
labelled_tweets.drop(labelled_tweets.loc[labelled_tweets['Sentiment'] == 'Neutral'].index, inplace = True)

tweets_token = labelled_tweets['Text'].apply(word_tokenize).tolist()

cleaned_tokens = []
for tokens in tweets_token:
    rm_noise = remove_noise(tokens, stop_words)
    lemma_tokens = lemmatize_sentence(rm_noise)
    cleaned_tokens.append(lemma_tokens)

new_tweet = []
for line in cleaned_tokens:
    line = ' '.join(line)
    new_tweet.append(line)
    

# Give a number based on type of response
def hate_label (row):
   if row['Sentiment'] == "Negative" :
      return 1
   if row['Sentiment'] == "Positive" :
      return 0
   return 'Other'

# Insert number as row
labelled_tweets['hate_label'] = labelled_tweets.apply (lambda row: hate_label(row), axis=1)



# examine the shape
labelled_tweets.shape

# examine the first 10 rows
labelled_tweets.head(10)
    

labelled_tweets.drop(['Sentiment'], axis = 1, inplace = True)
labelled_tweets.head(10)

# examine the class distribution
labelled_tweets.hate_label.value_counts()



# Generate wordcloud of common words
bully_words = ' '.join(list(labelled_tweets[labelled_tweets['hate_label'] == 1]['Text']))
bully_wc = WordCloud(width = 512,height = 380).generate(bully_words)
plt.figure(figsize = (10, 8), facecolor = (0, 0, 0))
plt.imshow(bully_wc)
plt.axis('off')
plt.tight_layout(pad = 0)
plt.show()

non_bully_words = ' '.join(list(labelled_tweets[labelled_tweets['hate_label'] == 0]['Text']))
non_bully_wc = WordCloud(width = 512,height = 380).generate(non_bully_words)
plt.figure(figsize = (10, 8), facecolor = 'k')
plt.imshow(non_bully_wc)
plt.axis('off')
plt.tight_layout(pad = 0)
plt.show()


## Data Splitting 
# how to define X and y (from the Tweeter data) for use with COUNTVECTORIZER
X = labelled_tweets.Text
y = labelled_tweets.hate_label
print(X.shape)
print(y.shape)

# split X and y into training and testing sets
from sklearn.model_selection import train_test_split

# split into 80% train dataset 
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.70, random_state=0)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


def tokenize(tweet):
    words = word_tokenize(tweet)
    
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    
    lemma = WordNetLemmatizer()
    words = [lemma.lemmatize(word) for word in words]
    
    return words
