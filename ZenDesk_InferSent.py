# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 17:24:30 2019

@author: masoudg
"""

import torch
import pandas as pd
import numpy as np
import os
import csv

# import some stuff
from sklearn.metrics import f1_score
from scipy import stats
from scipy.spatial import distance

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from ZenDesk_testModule import QA_FileParse

import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from models import InferSent

# data path and file name
data_path   = '//ad.monash.edu/home/User098/masoudg/Desktop/ZenDesk'
result_path = '//ad.monash.edu/home/User098/masoudg/Desktop/ZenDesk'
file_name   = 'zendesk_challenge.tsv'

# first lets read the tsv data file
with open(os.path.join(data_path, file_name)) as tsvfile:
    tsv_reader    = csv.reader(tsvfile, delimiter='\t')
    data_rows     = []
    for row in tsv_reader:
        data_rows.append(row)
        
_, _, answers, _, _, quetsions, labels = QA_FileParse(data_rows)


# load FB research InferSent model
model_version = 1
MODEL_PATH    = 'C:/MyFolder/InferSent/encoder/infersent%s.pkl' % model_version
params_model  = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': model_version}
infersent      = InferSent(params_model)
infersent.load_state_dict(torch.load(MODEL_PATH)) 

# Keep it on CPU or put it on GPU
use_cuda = False
infersent = infersent.cuda() if use_cuda else infersent

# If infersent1 model -> use GloVe embeddings. If infersent2 model -> use InferSent embeddings.
if model_version == 1:
    W2V_PATH = 'C:/MyFolder/InferSent/dataset/GloVe/glove.840B.300d.txt' 
else:
    W2V_PATH = 'C:/MyFolder/InferSent/dataset/fastText/crawl-300d-2M.vec' 
infersent.set_w2v_path(W2V_PATH)
# Load embeddings of K most frequent words
infersent.build_vocab_k_words(K=500000)
# or you can build your own vocabulary based on sentences in the data
#infersent.build_vocab(yoursentences, tokenize=True)

# 1- create sentence embedding for all the sentences and questions using InferSent 
# 2- calculates the distance between sentence & questions 
#    based on Euclidean & Cosine similarity using sentence embeddings
embeddings_dic  = {'Question':[],
                   'Answer':[],
                   'Question_Emb':[],
                   'Answer_Emb':[],
                   'Label':[],
                   'Cosine_Dist':[],
                   'Euclidean_Dist':[],
                   'Predicted_label_Cos':[],
                   'Predicted_label_Euc':[]}

pred_labels_cos = []
pred_labels_euc = []
for i_q, this_q in enumerate(quetsions):
    
    embeddings_q           = infersent.encode([this_q], tokenize=True, verbose=False)
    dist_cos_group         = []
    dist_euc_group         = []

    for i_a, this_a in enumerate(answers[i_q]):
        
        print(f'Question  {i_q: <10} Answer {i_a: <10} is done!')

        embeddings_a = infersent.encode([this_a], tokenize=True, verbose=False)
        
        # calculate the distances
        this_dist_cos = distance.cosine(embeddings_q, embeddings_a)
        this_dist_euc = distance.euclidean(embeddings_q, embeddings_a)
        dist_cos_group.append(this_dist_cos)
        dist_euc_group.append(this_dist_euc)
        
        # store the results in a dictionary
        embeddings_dic['Question'].append(this_q)
        embeddings_dic['Answer'].append(this_a)
        embeddings_dic['Question_Emb'].append(embeddings_q)
        embeddings_dic['Answer_Emb'].append(embeddings_a)
        embeddings_dic['Cosine_Dist'].append(this_dist_cos)
        embeddings_dic['Euclidean_Dist'].append(this_dist_euc)
        embeddings_dic['Label'].append(labels[i_q][i_a])
    
    # predict the labels
    ind                                            = np.argsort(dist_cos_group)
    pred_labels_groupd                             = np.zeros(len(labels[i_q]), dtype=int)
    pred_labels_groupd[ind[0:np.sum(labels[i_q])]] = 1
    pred_labels_cos.append(pred_labels_groupd)
    
    ind                                            = np.argsort(dist_euc_group)
    pred_labels_groupd                             = np.zeros(len(labels[i_q]), dtype=int)
    pred_labels_groupd[ind[0:np.sum(labels[i_q])]] = 1
    pred_labels_euc.append(pred_labels_groupd)
    
    

pred_labels_cos = [y for x in pred_labels_cos for y in x]
embeddings_dic['Predicted_label_Cos'].append(pred_labels_cos)

pred_labels_euc = [y for x in pred_labels_euc for y in x]
embeddings_dic['Predicted_label_Euc'].append(pred_labels_euc)

# make a datafarme
df             = pd.DataFrame(embeddings_dic)

# print some rows
print(df.head(5))

# We now calulate accuracy and f1 score
accuray_cos    = np.mean(df['Label'] == df['Predicted_label_Cos'])
accuray_euc    = np.mean(df['Label'] == df['Predicted_label_Euc'])

F1_score_cos   = f1_score(df['Label'], df['Predicted_label_Cos'])
F1_score_euc   = f1_score(df['Label'], df['Predicted_label_Cos'])

# print the results
print("Accuracy based on coise distance:        ", accuray_cos)
print("Accuracy based on euclidean distance:    ", accuray_euc)

print("F1 score based on coise distance:        ", F1_score_euc)
print("F1 score based on on euclidean distance: ", F1_score_euc)
        


        
