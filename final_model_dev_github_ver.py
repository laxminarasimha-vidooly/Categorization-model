# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 12:40:04 2019

@author: Hariom
"""
# =============================================================================
# Importing libraries
# =============================================================================
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.externals import joblib
import numpy as np
import string
from nltk.stem import PorterStemmer
ps = PorterStemmer()
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import urllib
import requests
import json
import nltk
import re
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import os, sys
import pickle
from sklearn.model_selection import GridSearchCV
os.chdir(Model for cat")

class categ:
#    def __init__(self, fnldf):
#        self.fnldf=fnldf
    def label_fact(self, fnldf):
        self.fnldf=fnldf
        self.fnldf = self.fnldf.iloc[:, 1:5]
        self.fnldf['category_id'] = self.fnldf['sub'].factorize()[0]
        category_id_df = self.fnldf[['sub', 'category_id']].drop_duplicates().sort_values('category_id')
        self.category_to_id = dict(category_id_df.values)
        self.id_to_category = dict(category_id_df[['category_id', 'sub']].values)

    def TFIDF(self):
        self.tfidf = TfidfVectorizer(sublinear_tf=True, min_df=22, norm='l2', encoding='latin-1', ngram_range=(1, 2))
        features = self.tfidf.fit_transform(self.fnldf.final_clean)
        labels = self.fnldf.category_id
        self.X2_train = features
        self.y2_train = labels.astype('int')
        filename1 = 'tfidf_final.sav'
#       Exporting TFIDF to joblib 
        joblib.dump(self.tfidf, open(filename1, 'wb'))
    
    def model_train(self):
        parameters = {'C': range(50,150,10), 'verbose': [1]}
        logreg = LogisticRegression()
        self.clf = GridSearchCV(logreg, parameters, cv=5)
        # clf.fit(scale_train_X, y2_train)
        self.clf.fit(self.X2_train, self.y2_train)
        
#        Exporting model to joblib
        filename = 'finalized_model_final.sav'
        joblib.dump(self.clf, open(filename, 'wb'))
        filename2 = 'category_to_id_final.sav'
        joblib.dump(self.category_to_id, open(filename2, 'wb'))
        filename3 = 'id_to_category_final.sav'
        joblib.dump(self.id_to_category, open(filename3, 'wb'))
    
    def test_data(self,testdf):
        self.label_fact(testdf)
        filename1 = 'tfidf_final.sav'
        tfidf = joblib.load(open(filename1, 'rb'))

        self.X2_test=tfidf.transform(testdf.final_clean)
               
    def model_test(self):
        filename = 'finalized_model_final.sav'
        
#        filename2 = 'category_to_id_final.sav'
#        filename3 = 'id_to_category_final.sav'
        self.clf=joblib.load(open(filename, 'rb'))
#        
#        self.category_to_id = joblib.load(open(filename2, 'rb'))
#        self.id_to_category = joblib.load(open(filename3, 'rb'))

        y_pred = self.clf.predict(self.X2_test)
        s1 = []
        
        for i in range(len(self.fnldf)):
    #     s1.append(id_to_category[y1_pred[i]])
            s1.append(self.category_to_id[self.fnldf.iloc[i, 3]])
        self.fnldf['actal_cat_id'] = s1
        
        self.fnldf['pred_cat_id'] = y_pred

        s2 = []
        for i in range(len(y_pred)):
    #     s1.append(id_to_category[y1_pred[i]])
            s2.append(self.id_to_category[y_pred[i]])
        self.fnldf['pred_sub'] = s2
        
        self.fnldf.to_csv("test_pred_subcat_logistic.csv", sep='\t', encoding='utf-8')
        
        accuracy1 = accuracy_score(self.fnldf["actal_cat_id"], self.fnldf["pred_cat_id"])
        print("accuracy of the model is", accuracy1)
        cv_results = pd.DataFrame(self.clf.cv_results_)
        
        plt.figure(figsize=(8, 6))
        plt.plot(cv_results['param_C'], cv_results['mean_test_score'])
        plt.plot(cv_results['param_C'], cv_results['mean_train_score'])
        plt.xlabel('C')
        plt.ylabel('Accuracy')
        plt.legend(['test accuracy', 'train accuracy'], loc='upper left')
        # plt.xscale('log')
        
fnldf= pd.read_csv("train_5.csv")
mod_cat=categ()
mod_cat.label_fact(fnldf)
mod_cat.TFIDF()
mod_cat.model_train()
fnldf= pd.read_csv("test_5.csv")
mod_cat.test_data(fnldf)
mod_cat.model_test()
        
        

        
        