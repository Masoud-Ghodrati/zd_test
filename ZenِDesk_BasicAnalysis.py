# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 11:56:41 2019

@author: masoudg
"""

# import some stuff 
import numpy as np
import os
import csv

import nltk
from nltk.corpus import stopwords
stopword    = stopwords.words('english')

from sklearn.metrics import f1_score
from scipy import stats

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

from ZenDesk_testModule import QA_FileParse

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
answers_word, answers_word_len, answers, questions_word, questions_word_len, questions, labels = QA_FileParse(data_rows)

# First, lets consider a simple word matching method, so we count the number
# of non-stopwords in the question that also occur in the answer sentence
            
match_word      = []
pred_labels     = []
pred_labels_max = []
for i_q, this_q in enumerate(questions_word):
    this_match_word = []
    for i_d, this_a in enumerate(answers_word[i_q]):
        this_match_word.append(len(set(this_q).intersection(this_a))) # bacis matching of two lists  
      
    match_word.append(this_match_word)
    pred_labels.append(np.array(this_match_word) > 0)
    pred_labels_max.append(np.array(this_match_word) == max(this_match_word))

# We now flatten the guessed label based on "matching word" method to calculate accuracy
flattened_labels          = np.array([y for x in labels for y in x])
flattened_pred_labels     = np.array([y for x in pred_labels for y in x], dtype=int)
flattened_pred_labels_max = np.array([y for x in pred_labels_max for y in x], dtype=int)

# We now calulate accuracy and f1 score
accuray_pred_labels       = np.mean(flattened_pred_labels == flattened_labels)
accuray_pred_labels_max   = np.mean(flattened_pred_labels_max == flattened_labels)

F1_score_pred_labels      = f1_score(flattened_labels, flattened_pred_labels)
F1_score_pred_labels_max  = f1_score(flattened_labels, flattened_pred_labels_max)

print("Accuracy based on the number of matched words in Q and A:     ", accuray_pred_labels)
print("Accuracy based on the max number of matched words in Q and A: ", accuray_pred_labels_max)

print("F1 score based on the number of matched words in Q and A:     ", F1_score_pred_labels)
print("F1 score based on the max number of matched words in Q and A: ", F1_score_pred_labels_max)

# we now make a random distribution of labels to see if word matching does a good job
n_rand_sample            = 1000          
rand_label_dist_rnd1_acc = []
rand_label_dist_rnd1_F1  = []
rand_labels              = np.arange(len(flattened_labels))
for i in range(n_rand_sample):
    
    np.random.shuffle(rand_labels)
    rand_label_dist_rnd1_acc.append(np.mean( flattened_labels[rand_labels] == flattened_labels))
    rand_label_dist_rnd1_F1.append(f1_score( flattened_labels[rand_labels],   flattened_labels))

# Note that there are two ways to randomize, 1) randomize the whole list of labels
# like the one we did, or 2) ranomize within each doc/answer list. Now, we the 
# the second method
rand_label_dist_rnd2_acc = []
rand_label_dist_rnd2_F1  = []
for i in range(n_rand_sample):
    this_rand_label           = []
    this_flattened_rand_label = []
    for l in labels:
        rand_labels = np.arange(len(l))
        np.random.shuffle(rand_labels)
        this_l = np.array(l)
        this_rand_label.append(this_l[rand_labels])
    
    this_flattened_rand_label = [y for x in this_rand_label for y in x]
    rand_label_dist_rnd2_acc.append(np.mean( this_flattened_rand_label == flattened_labels))
    rand_label_dist_rnd2_F1.append(f1_score( this_flattened_rand_label,   flattened_labels))

# Now we check the word/category frequency to see if we can get any information 
# out of Questions or Answers
flattened_questions_word = [y for x in questions_word for y in x]
flattened_answers_word   = [z for y in answers_word for x in y for z in x]  
# nltk.FreqDist generates a tally of the number of times each word appears
# and stores the results in a special dictionary.
word_count_q             = nltk.FreqDist(flattened_questions_word)
word_count_a             = nltk.FreqDist(flattened_answers_word)  

# Sort the dictionary to find the most and least common terms.
# Returns a list.
sorted_word_count_q = [(word_count_q[key], key) for key in word_count_q]
sorted_word_count_q.sort()
sorted_word_count_q.reverse()
    
sorted_word_count_a = [(word_count_a[key], key) for key in word_count_a]
sorted_word_count_a.sort()
sorted_word_count_a.reverse()

print(f'\n\n|  Q word  |   count  |  A word  |   count  |')
print(f'|----------|----------|----------|----------|')
for i in range(10):
    print(f'|{sorted_word_count_q[i][1]: ^10}|{sorted_word_count_q[i][0]: ^10}|{sorted_word_count_a[i][1]: ^10}|{sorted_word_count_a[i][0]: ^10}')
    print(f'|----------|----------|----------|----------|')    

    
# We now do a very basic investigation to see if 1) the length of a question 
# is correlated with answer 2) correct answers have longer lengths
corr_sen_len   = []
incorr_sen_len = []
for i1, l1 in enumerate(labels):
    for i2, l2 in enumerate(l1):
        if l2 == 1:
            corr_sen_len.append([questions_word_len[i1], answers_word_len[i1][i2]])
        else:
            incorr_sen_len.append([questions_word_len[i1], answers_word_len[i1][i2]])
            
corr_sen_len   = np.array(corr_sen_len)   
incorr_sen_len = np.array(incorr_sen_len)                
            
    

# now we visualize the results  

# F1 score results for word marching 
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 16}	
plt.rc('font', **font)
plt.rc('xtick', labelsize=16)     
plt.rc('ytick', labelsize=16)
nBins = 50    
plt.figure(figsize=(15, 7))
ax1 = plt.subplot(121)
plt.hist(rand_label_dist_rnd1_F1, bins=nBins, density=True, facecolor='g', edgecolor='g', label="Distribution of randomized labels F1 score")
plt.plot(F1_score_pred_labels_max*np.ones(2), [0, 100], '--r', label="F1 score of matching word method")
plt.xlabel('F1 Score')
plt.ylabel('Frequency')
plt.title('randomization 1')
plt.axis([0.05, 0.4, 0, 60])
plt.legend(frameon=False)

ax2 = plt.subplot(122, sharex=ax1, sharey=ax1)
plt.hist(rand_label_dist_rnd2_F1, bins=nBins, density=True, facecolor='g', edgecolor='g')
plt.plot(F1_score_pred_labels_max*np.ones(2), [0, 100], '--r')
plt.xlabel('F1 Score')
plt.ylabel('Frequency')
plt.title('randomization 2')
plt.axis([0.05, 0.4, 0, 60])

plt.suptitle('The performance of word matching')
plt.savefig(result_path + "/"  + "Word_Matching_Performance", dpi=300)
plt.show()

# correlction results between Q and A lengths
plt.figure(figsize=(15, 7))
ax1 = plt.subplot(121)
sns.regplot(x=corr_sen_len[:,0], y=corr_sen_len[:,1], color="g", ax=ax1)
corr = stats.pearsonr(x=corr_sen_len[:,0], y=corr_sen_len[:,1])
plt.text(1, 90,"Pearson r = " + str(corr[0]))
plt.text(1, 85,"p = " + str(corr[1]))
plt.xlabel('Question length (#non-stop words)')
plt.ylabel('Correct answer length (#non-stop words)')
plt.axis([0, 20, 0, 100])


ax2 = plt.subplot(122)
sns.regplot(x=incorr_sen_len[:,0], y=incorr_sen_len[:,1], color="g", ax=ax2)
corr = stats.pearsonr(x=incorr_sen_len[:,0], y=incorr_sen_len[:,1])
plt.text(1, 90,"Pearson r = " + str(corr[0]))
plt.text(1, 85,"p = " + str(corr[1]))
plt.xlabel('Question length (#non-stop words)')
plt.ylabel('Incorrect answer length (#non-stop words)')
plt.axis([0, 20, 0, 100])

plt.suptitle('The correlation between question and answer length')
plt.savefig(result_path + "/"  + "QA_length_corr", dpi=300)
plt.show()

# Word Frequency results in a world could
plt.figure(figsize=(25, 14))
x, y      = np.ogrid[:300, :300]
mask      = (x - 150) ** 2 + (y - 150) ** 2 > 130 ** 2
mask      = 255 * mask.astype(int)
max_Words = 500

ax1 = plt.subplot(121)
wordcloud = WordCloud(stopwords=stopword, collocations=False, max_font_size=80, background_color="white",
                      repeat=True, mask=mask, max_words = max_Words)
wordcloud.generate(" ".join([(k + ' ') * v for k,v in word_count_q.items()]))
plt.imshow(wordcloud, interpolation="bilinear")
plt.title("Word Freq./Occur. in Questions")
plt.axis("off")

ax1 = plt.subplot(122)
wordcloud = WordCloud(stopwords=stopword,collocations=False, max_font_size=80, background_color="white",
                      repeat=True,mask=mask, max_words = max_Words + 500)
wordcloud.generate(" ".join([(k + ' ') * v for k,v in word_count_a.items()]))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Freq./Occur. in Answers")

plt.suptitle('Word Frequency visualization')
plt.savefig(result_path + "/"  + "Word_Frequency", dpi=300)
plt.show()








            
