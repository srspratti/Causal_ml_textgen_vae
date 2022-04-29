import json
import os
import torch
import pandas as pd
import numpy as np
import spacy
import sklearn.metrics

from os import listdir
from os.path import isfile, join
from model import SentenceVAE
from utils import to_var, idx2word, interpolate
from sklearn.preprocessing import OneHotEncoder
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score

analyser = SentimentIntensityAnalyzer()

nlp = spacy.load("en_core_web_sm")

def sentiment_analyzer_scores(sentence):
    score = analyser.polarity_scores(sentence)
    if score['compound']> 0.05:
        # return 1, 'Positive'
        return [1, 0]
    else:
        # return 0, 'Negative'
        return [0, 1]


# yelp_train = pd.read_csv("data/yelp_valid.csv")
# yelp_generated = pd.read_csv("data/yelp_generated.csv")
yelp_generated_x = pd.read_csv("data/yelp_generated.csv") #yelp_generated['text'] 
yelp_generated_y = pd.read_csv("data/y_yelp_generated.csv") #yelp_generated['labels']



'''
METHODS TO LABELING THE SENTENCES AS PLURAL, SINGULAR OR NEUTRAL

'''

def pos(sentence):

    '''
    Pos labels one single sentences by using Part-Of-Speech Tagging
    :param sentence: (STR)
    :return:
    '''
    sentence = sent_tokenize(sentence)[0]
    doc = nlp(sentence)
    singular = 0
    plural = 0
    neutral = 0
    for idx, token in enumerate(doc):
        if token.tag_ == "NN" or str(token).lower() in ["i","he","she", "it", "myself"]:
            singular+=1
        if token.tag_ == "NNS" or str(token).lower() in ["we", "they", "themselves", "themself", "ourselves", "ourself"]:
            plural+=1
    if singular > plural:
        #print("SINGULAR: ",sentence)
        return [1,0,0]
    if singular < plural:
        #print("PLURAL: ",sentence)
        return [0,0,1]
    if singular == plural:
        #print("NEUTRAL: ",sentence)
        return [0,1,0]

labels = []
for idx, s in enumerate(yelp_generated_x.text):
    # print(idx)
    # print("s: ", s)
    label_sent = sentiment_analyzer_scores(s)
    label = pos(s)
    label_sent.extend(label)
    # labels.append(label)
    labels.append(label_sent)

labels = np.asarray(labels)

# labels = np.asarray(labels)

# labels = pd.DataFrame({'Singular': labels[:,0],
#                        'Neutral': labels[:,1],
#                        'Plural': labels[:,2]})

labels_df = pd.DataFrame({'Positive': labels[:,0],
                        'Negative': labels[:,1],
                       'Singular': labels[:,2],
                       'Neutral': labels[:,3],
                       'Plural': labels[:,4]})

name = "calculated_sent_pronoun"

labels_df.to_csv("data/y_yelp_"+name+".csv", index=False)

#----------------------------------------------Calculate the accuracy for multi-label classification 

"""

yelp_generated_y = y_true
labels = y_pred

"""

# ----------Reference : https://stats.stackexchange.com/questions/233275/multilabel-classification-metrics-on-scikit 

def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    '''
    Compute the Hamming score (a.k.a. label-based accuracy) for the multi-label case
    https://stackoverflow.com/q/32239577/395857
    '''
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        #print('\nset_true: {0}'.format(set_true))
        #print('set_pred: {0}'.format(set_pred))
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/\
                    float( len(set_true.union(set_pred)) )
        #print('tmp_a: {0}'.format(tmp_a))
        acc_list.append(tmp_a)
    return np.mean(acc_list)



y_true = yelp_generated_y.copy() 
y_pred = labels.copy()
y_true = np.asarray(y_true)

print('Hamming score: {0}'.format(hamming_score(y_true, y_pred))) # 0.375 (= (0.5+1+0+0)/4)

# For comparison sake:

# Subset accuracy
# 0.25 (= 0+1+0+0 / 4) --> 1 if the prediction for one sample fully matches the gold. 0 otherwise.
print('Subset accuracy: {0}'.format(sklearn.metrics.accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)))

print('Hamming loss: {0}'.format(sklearn.metrics.hamming_loss(y_true, y_pred))) 