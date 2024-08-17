# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 19:04:13 2022

@author: asus
"""

import numpy as np
from colorama import Fore, Style
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, f1_score, recall_score

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
