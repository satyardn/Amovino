#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import sys
import seaborn as sns
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.tokenize import RegexpTokenizer

# In[4]:


wine_df = pd.read_csv("C://Users/LENOVO/Google Drive/MSc in Computing/CA675 Cloud Technologies/Assignment 2/wine-reviews/winemag-data.csv")


# In[5]:


points_list = wine_df['points'].tolist()



number_list = list(range(1,49))
country_list=wine_df['country'].unique()
country_dict = dict(zip(country_list,number_list))


new_df = wine_df.copy()
new_df.replace(country_dict,inplace=True)
new_df = new_df[new_df['country'].apply(lambda x: isinstance(x, (int, np.int64)))]
export_csv = new_df.to_csv (r'C://Users/LENOVO/Google Drive/MSc in Computing/export_dataframe.csv', index = None, header=True)


# In[6]:


variety_list = wine_df['variety'].unique()
num_list = list(range(50,700))
variety_dict = dict(zip(variety_list,num_list))
new_df.replace(variety_dict,inplace=True)
new_df = new_df[new_df['variety'].apply(lambda x: isinstance(x, (int, np.int64)))]
export_csv = new_df.to_csv (r'C://Users/LENOVO/Google Drive/MSc in Computing/new_export_dataframe.csv', index = None, header=True)


# In[7]:


df = pd.read_csv('C://Users/LENOVO/Google Drive/MSc in Computing/new_export_dataframe.csv')


# In[8]:


TRAIN_SET_COUNT = len(df['points'])
# print(TRAIN_SET_COUNT)

TRAIN_INPUT = list()
TRAIN_OUTPUT = list()
for i in range(TRAIN_SET_COUNT):
    a = df.loc[i,'points']
    b =  df.loc[i,'country']
    c = df.loc[i,'variety']
    op = df.loc[i,'price']
    TRAIN_INPUT.append([a,b,c])
    TRAIN_OUTPUT.append(op)


# In[9]:


from sklearn.linear_model import LinearRegression

predictor = LinearRegression()
predictor.fit(X=TRAIN_INPUT, y=TRAIN_OUTPUT)


# In[13]:

a, b, c = (int(x) for x in sys.argv[1:])
X_TEST = [[a,b,c]]
outcome = predictor.predict(X=X_TEST)
coefficients = predictor.coef_

print('Outcome : {}\nCoefficients : {}'.format(outcome, coefficients))
f=open("C://Users/LENOVO/Google Drive/MSc in Computing/temp.txt","w")
f.write(np.array_str(outcome))
f.close()