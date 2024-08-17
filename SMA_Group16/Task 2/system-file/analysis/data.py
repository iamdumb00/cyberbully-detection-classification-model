# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 19:07:00 2022

@author: asus
"""
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


# Analysis #

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
