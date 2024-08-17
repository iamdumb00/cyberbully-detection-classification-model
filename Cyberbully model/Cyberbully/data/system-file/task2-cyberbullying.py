# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 17:11:49 2022

@author: asus
"""


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
# %matplotlib inline
from wordcloud import WordCloud
from math import log, sqrt
import pandas as pd
import numpy as np
import re
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


#----Model Development Section----# 

## Bag of Words Transformation 

# instantiate the vectorizer
vect = CountVectorizer(analyzer='word', binary=False, decode_error='strict',
        encoding='ISO-8859-1', input='content',
        lowercase=True, max_df=1.0, max_features=None, min_df=1,
        ngram_range=(1, 2), preprocessor=None, stop_words='english',
        strip_accents=None, token_pattern='(?u)\\b\\w\\w+\\b',
        tokenizer=tokenize, vocabulary=None)


# equivalently: combine fit and transform into a single step
X_train_dtm = vect.fit_transform(X_train)

# examine the document-term matrix
X_train_dtm

# transform testing data (using fitted vocabulary) into a document-term matrix
X_test_dtm = vect.transform(X_test)
print(X_test_dtm)


## 1. DECISION TREE CLASSIFIER 

# train the model
from sklearn import tree
dc = tree.DecisionTreeClassifier(criterion='entropy')
%time dc.fit(X_train_dtm, y_train)

# Check for model performance/evaluation
# Accuracy can be computed by comparing test values (y_test) and predicted values (y_pred_class)
y_pred_class = dc.predict(X_test_dtm)

from sklearn import metrics
from sklearn.metrics import confusion_matrix
metrics.accuracy_score(y_test, y_pred_class)
cf_matrix = confusion_matrix(y_test, y_pred_class)

# print the classification report
print("Decision Tree evaluation metrics: ", metrics.classification_report(y_test, y_pred_class))


## 2. LOGISTIC REGRESSION CLASSIFIER

# import and instantiate a logistic regression model
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

# train the model using X_train_dtm
%time lr.fit(X_train_dtm, y_train)

# make class predictions for X_test_dtm
y_pred_class = lr.predict(X_test_dtm)

# calculate predicted probabilities for X_test_dtm (well calibrated)
y_pred_prob = lr.predict_proba(X_test_dtm)[:, 1]

# calculate accuracy
from sklearn import metrics
from sklearn.metrics import confusion_matrix
metrics.accuracy_score(y_test, y_pred_class)
cf_matrix = confusion_matrix(y_test, y_pred_class)

# print the classification report
print("Logistic Regression evaluation metrics: ", metrics.classification_report(y_test, y_pred_class))



## 3. NAIVE BAYES CLASSIFIER 
# import and instantiate a Multinomial Naive Bayes model
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()

# train the model using X_train_dtm (timing it with an IPython "magic command")
%time nb.fit(X_train_dtm, y_train)

# make class predictions for X_test_dtm
y_pred_class = nb.predict(X_test_dtm)

# calculate accuracy of class predictions
metrics.accuracy_score(y_test, y_pred_class)

# print the confusion matrix
cf_matrix = confusion_matrix(y_test, y_pred_class)

# print the classification report
print("Naive Bayes evaluation metrics: ", metrics.classification_report(y_test, y_pred_class))


## 4. KNN CLASSIFIER 

# train the model
from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier(weights='distance', n_neighbors=2)  
%time knn.fit(X_train_dtm, y_train)

# Check for model performance/evaluation
# Accuracy can be computed by comparing test values (y_test) and predicted values (y_pred_class)
y_pred_class = knn.predict(X_test_dtm)
metrics.accuracy_score(y_test, y_pred_class)

cf_matrix = confusion_matrix(y_test, y_pred_class)

# print the classification report
print("KNN evaluation metrics: ", metrics.classification_report(y_test, y_pred_class))



## PLOT CONFUSION MATRIX 
  
    # Decision Tree
      def model_evaluation(dc, y_test, y_pred_class):
      
        print(
          f'{Fore.YELLOW}{dc}{Style.RESET_ALL}'
          )
      
        ## Confusion Matrix
        cf_matrix = confusion_matrix(y_test, y_pred_class)
      
        # Plot Confusion Matrix
        print(
          f'{Fore.MAGENTA}confusion_matrix{Style.RESET_ALL}\n'
          f'{cf_matrix}'
          )
       
      group_names = ['True Negatuve','False Positive',
                     'False Negative','True Positive']
      
      group_counts = ["{0:0.0f}".format(value) for value in
                      cf_matrix.flatten()]
      
      group_percentages = ["{0:.2%}".format(value) for value in
                           cf_matrix.flatten()/np.sum(cf_matrix)]
      
      labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
                zip(group_names,group_counts,group_percentages)]
      
      labels = np.asarray(labels).reshape(2,2)
      
      ax = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
      
      ax.set_xlabel('\nPredicted Values')
      ax.set_ylabel('Actual Values ');
      
      ## Ticket labels - List must be in alphabetical order
      ax.xaxis.set_ticklabels(['0','1'])
      ax.yaxis.set_ticklabels(['0','1'])
      
      ## Display the visualization of the Confusion Matrix.
      plt.show()
      
      ## DC: Classification Report & Accuracy Score
      print(
      f'{Fore.YELLOW}{dc}{Style.RESET_ALL}'
      f'{Fore.MAGENTA}Classification Report{Style.RESET_ALL}\n'
      f'{classification_report(y_test, y_pred_class)}\n'
      f'{Fore.MAGENTA}Accuracy Score{Style.RESET_ALL}: {accuracy_score(y_test, y_pred_class)}\n'
      f'{Fore.MAGENTA}Precision Score{Style.RESET_ALL}: {precision_score(y_test, y_pred_class)}\n'
      f'{Fore.MAGENTA}Recall Score{Style.RESET_ALL}: {recall_score(y_test, y_pred_class)}\n'
      f'{Fore.MAGENTA}F1 Score{Style.RESET_ALL}: {f1_score(y_test, y_pred_class)}'
      )
  
    
    # Logistic Regression 
    def model_evaluation(lr, y_test, y_pred_class):
    
      print(
        f'{Fore.YELLOW}{lr}{Style.RESET_ALL}'
        )
    
      ## Confusion Matrix
      cf_matrix = confusion_matrix(y_test, y_pred_class)
    
      # Plot Confusion Matrix
      print(
        f'{Fore.MAGENTA}confusion_matrix{Style.RESET_ALL}\n'
        f'{cf_matrix}'
        )
     
    group_names = ['True Negatuve','False Positive',
                   'False Negative','True Positive']
    
    group_counts = ["{0:0.0f}".format(value) for value in
                    cf_matrix.flatten()]
    
    group_percentages = ["{0:.2%}".format(value) for value in
                         cf_matrix.flatten()/np.sum(cf_matrix)]
    
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
              zip(group_names,group_counts,group_percentages)]
    
    labels = np.asarray(labels).reshape(2,2)
    
    ax = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
    
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ');
    
    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['0','1'])
    ax.yaxis.set_ticklabels(['0','1'])
    
    ## Display the visualization of the Confusion Matrix.
    plt.show()
    
    ## LR: Classification Report & Accuracy Score
    print(
    f'{Fore.YELLOW}{lr}{Style.RESET_ALL}'
    f'{Fore.MAGENTA}Classification Report{Style.RESET_ALL}\n'
    f'{classification_report(y_test, y_pred_class)}\n'
    f'{Fore.MAGENTA}Accuracy Score{Style.RESET_ALL}: {accuracy_score(y_test, y_pred_class)}\n'
    f'{Fore.MAGENTA}Precision Score{Style.RESET_ALL}: {precision_score(y_test, y_pred_class)}\n'
    f'{Fore.MAGENTA}Recall Score{Style.RESET_ALL}: {recall_score(y_test, y_pred_class)}\n'
    f'{Fore.MAGENTA}F1 Score{Style.RESET_ALL}: {f1_score(y_test, y_pred_class)}'
    )
 
    
   # Naive Bayes
   def model_evaluation(nb, y_test, y_pred_class):
   
     print(
       f'{Fore.YELLOW}{nb}{Style.RESET_ALL}'
       )
   
     ## Confusion Matrix
     cf_matrix = confusion_matrix(y_test, y_pred_class)
   
     # Plot Confusion Matrix
     print(
       f'{Fore.MAGENTA}confusion_matrix{Style.RESET_ALL}\n'
       f'{cf_matrix}'
       )
    
   group_names = ['True Negatuve','False Positive',
                  'False Negative','True Positive']
   
   group_counts = ["{0:0.0f}".format(value) for value in
                   cf_matrix.flatten()]
   
   group_percentages = ["{0:.2%}".format(value) for value in
                        cf_matrix.flatten()/np.sum(cf_matrix)]
   
   labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
             zip(group_names,group_counts,group_percentages)]
   
   labels = np.asarray(labels).reshape(2,2)
   
   ax = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
   
   ax.set_xlabel('\nPredicted Values')
   ax.set_ylabel('Actual Values ');
   
   ## Ticket labels - List must be in alphabetical order
   ax.xaxis.set_ticklabels(['0','1'])
   ax.yaxis.set_ticklabels(['0','1'])
   
   ## Display the visualization of the Confusion Matrix.
   plt.show()
   
   ## NB: Classification Report & Accuracy Score
   print(
   f'{Fore.YELLOW}{nb}{Style.RESET_ALL}'
   f'{Fore.MAGENTA}Classification Report{Style.RESET_ALL}\n'
   f'{classification_report(y_test, y_pred_class)}\n'
   f'{Fore.MAGENTA}Accuracy Score{Style.RESET_ALL}: {accuracy_score(y_test, y_pred_class)}\n'
   f'{Fore.MAGENTA}Precision Score{Style.RESET_ALL}: {precision_score(y_test, y_pred_class)}\n'
   f'{Fore.MAGENTA}Recall Score{Style.RESET_ALL}: {recall_score(y_test, y_pred_class)}\n'
   f'{Fore.MAGENTA}F1 Score{Style.RESET_ALL}: {f1_score(y_test, y_pred_class)}'
   )
   

    # KNN
    def model_evaluation(knn, y_test, y_pred_class):
    
      print(
        f'{Fore.YELLOW}{knn}{Style.RESET_ALL}'
        )
    
      ## Confusion Matrix
      cf_matrix = confusion_matrix(y_test, y_pred_class)
    
      # Plot Confusion Matrix
      print(
        f'{Fore.MAGENTA}confusion_matrix{Style.RESET_ALL}\n'
        f'{cf_matrix}'
        )
     
    group_names = ['True Negatuve','False Positive',
                   'False Negative','True Positive']
    
    group_counts = ["{0:0.0f}".format(value) for value in
                    cf_matrix.flatten()]
    
    group_percentages = ["{0:.2%}".format(value) for value in
                         cf_matrix.flatten()/np.sum(cf_matrix)]
    
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
              zip(group_names,group_counts,group_percentages)]
    
    labels = np.asarray(labels).reshape(2,2)
    
    ax = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
    
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ');
    
    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['0','1'])
    ax.yaxis.set_ticklabels(['0','1'])
    
    ## Display the visualization of the Confusion Matrix.
    plt.show()
    
    ## KNN: Classification Report & Accuracy Score
      print(
        f'{Fore.MAGENTA}Classification Report{Style.RESET_ALL}\n'
        f'{classification_report(y_test, y_pred_class)}\n'
        f'{Fore.MAGENTA}Accuracy Score{Style.RESET_ALL}: {accuracy_score(y_test, y_pred_class)}\n'
        f'{Fore.MAGENTA}Precision Score{Style.RESET_ALL}: {precision_score(y_test, y_pred_class)}\n'
        f'{Fore.MAGENTA}Recall Score{Style.RESET_ALL}: {recall_score(y_test, y_pred_class)}\n'
        f'{Fore.MAGENTA}F1 Score{Style.RESET_ALL}: {f1_score(y_test, y_pred_class)}'
        ) 
     




















