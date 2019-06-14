# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 19:21:53 2019

@author: masoudg
"""

# import some stuff
import numpy as np
import csv
import os

from sklearn.metrics import f1_score
import spacy
en_nlp = spacy.load('en_core_web_sm')
from nltk.stem.lancaster import LancasterStemmer
st = LancasterStemmer()

from ZenDesk_testModule import QA_FileParse

# data path and file name
data_path   = '.../...'
result_path = '.../...'
file_name   = 'zendesk_challenge.tsv'

# first lets read the tsv data file
with open(os.path.join(data_path, file_name)) as tsvfile:
    tsv_reader    = csv.reader(tsvfile, delimiter='\t')
    data_rows     = []
    for row in tsv_reader:
        data_rows.append(row)
        
_, _, answers, _, _, quetsions, labels = QA_FileParse(data_rows)


# predict labels
pred_labels     = []
for i_q, this_q in enumerate(quetsions):
    
    this_q      = this_q.lower()   
    this_q_root = st.stem(str([sent.root for sent in en_nlp(this_q).sents][0]))
    pred_labels_group = []
    for i_a, this_a in enumerate(answers[i_q]):
                
        this_a_root = [st.stem(chunk.root.head.text.lower()) for chunk in en_nlp(this_a).noun_chunks]
    
        if this_q_root in this_a_root: 
            pred_labels_group.append(1)
        else:
            pred_labels_group.append(0)
    
    pred_labels.append(pred_labels_group)
    
    
            
# We now flatten the guessed label based on "matching word" method to calculate accuracy
flattened_labels          = np.array([y for x in labels for y in x])
flattened_pred_labels     = np.array([y for x in pred_labels for y in x], dtype=int)

# We now calulate accuracy and f1 score
accuray_pred_labels       = np.mean(flattened_pred_labels == flattened_labels)
F1_score_pred_labels      = f1_score(flattened_labels, flattened_pred_labels)

print("Accuracy based on the number of matched roots in Q and A:     ", accuray_pred_labels)
print("F1 score based on the number of matched root in Q and A:     ", F1_score_pred_labels)









