# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 09:55:37 2019

@author: masoudg
"""
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def QA_FileParse(data_rows):
    """This file reads the tsv file and returns a bunch of lists that contain the 
    information for questions_word, answers_word, and Labels 
    
    A datapoint in this dataset has a query, a document and thier relevance(0: irrelevant, 1: relevant)
    
    Example data point:
    QuestionID	Question	DocumentID	DocumentTitle	SentenceID	Sentence	Label
    Q1	how are glacier caves formed?	D1	Glacier cave	D1-0	A partly submerged glacier cave on Perito Moreno Glacier .	0

    """    
    # Defining some consants for .tsv reading
    # These refer to the column indexes of certain data
    QUESTION_ID_INDEX = 0
    QUESTION_INDEX    = 1
    ANSWER_INDEX      = 5
    LABEL_INDEX       = 6
    
    # grouping all answers and labels which belong to one question into
    # one group. 
    answer_word_group     = []
    answer_word_group_len = []
    answer_group          = []
    label_group           = []
    
    
    # counting number of documents so we can remove those question-answer pairs
    # which do not have even one relevant document
    n_relevant_ans  = 0
    n_filtered_ans  = 0
    
    questions_word     = []
    questions_word_len = []
    answers_word       = []
    answers_word_len   = []
    questions          = []
    answers            = []
    labels             = []
    
    # lets lemmatize and and remove stop words from questions and answers
    # we load them from NLTK 
    stopwordsList      = stopwords.words('english')
    other_stopwords    = ['dont','didnt','doesnt','cant','couldnt','couldve','',
                       'im','ive','isnt','theres','wasnt','wouldnt','a','what',
                       'how','why','where','many', 'much']
    for i_sw in other_stopwords:
        stopwordsList.append(i_sw)

    wordnet_lemmatizer = WordNetLemmatizer()
    
    for i, line in enumerate(data_rows[1:], start=1):
        if i < len(data_rows) - 1:  # check if out of bounds might occur
            # If the question id index doesn't change
            if data_rows[i][QUESTION_ID_INDEX] == data_rows[i + 1][QUESTION_ID_INDEX]:
                
                rawText                 = data_rows[i][ANSWER_INDEX].lower().replace("'", "")
                word_tokens             = re.sub("[^a-zA-Z0-9]", " ", rawText).split()                
                removing_stopwordsLists = [word for word in word_tokens if word not in stopwordsList]
                lemmatized_word         = [wordnet_lemmatizer.lemmatize(word) for word in removing_stopwordsLists]
                answer_word_group.append(lemmatized_word)
                answer_word_group_len.append(len(removing_stopwordsLists))
                answer_group.append(rawText)
                label_group.append(int(data_rows[i][LABEL_INDEX]))
                
                
                n_relevant_ans         += int(data_rows[i][LABEL_INDEX])
            else:
                rawText                 = data_rows[i][ANSWER_INDEX].lower().replace("'", "")
                word_tokens             = re.sub("[^a-zA-Z0-9]", " ", rawText).split() 
                removing_stopwordsLists = [word for word in word_tokens if word not in stopwordsList]
                lemmatized_word         = [wordnet_lemmatizer.lemmatize(word) for word in removing_stopwordsLists]
                answer_word_group.append(lemmatized_word)
                answer_word_group_len.append(len(removing_stopwordsLists))
                answer_group.append(rawText)
                label_group.append(int(data_rows[i][LABEL_INDEX]))
                
                n_relevant_ans         += int(data_rows[i][LABEL_INDEX])
    
                if n_relevant_ans > 0:
                    
                    answers_word.append(answer_word_group)
                    answers_word_len.append(answer_word_group_len)
                    labels.append(label_group)
                    answers.append(answer_group)
                    
                    rawText                 = data_rows[i][QUESTION_INDEX].lower().replace("'", "")
                    word_tokens             = re.sub("[^a-zA-Z0-9]", " ", rawText).split() 
                    removing_stopwordsLists = [word for word in word_tokens if word not in stopwordsList]
                    lemmatized_word         = [wordnet_lemmatizer.lemmatize(word) for word in removing_stopwordsLists]
                    questions_word.append(lemmatized_word)
                    questions_word_len.append(len(removing_stopwordsLists))
                    questions.append(rawText)
                    
                   
                else:
                    # Filter out a question if it doesn't have a single relevant answer
                    n_filtered_ans         += 1
    
                n_relevant_ans        = 0
                answer_word_group     = []
                answer_group          = []
                answer_word_group_len = []
                label_group           = []
    
        else:
            # If we are on the last line
            rawText                 = data_rows[i][ANSWER_INDEX].lower().replace("'", "")
            word_tokens             = re.sub("[^a-zA-Z0-9]", " ", rawText).split() 
            removing_stopwordsLists = [word for word in word_tokens if word not in stopwordsList]
            lemmatized_word         = [wordnet_lemmatizer.lemmatize(word) for word in removing_stopwordsLists]
            answer_word_group.append(lemmatized_word) 
            answer_word_group_len.append(len(removing_stopwordsLists))
            label_group.append(int(data_rows[i][LABEL_INDEX]))
            answer_group.append(rawText)
            
            n_relevant_ans         += int(data_rows[i][LABEL_INDEX])
    
            if n_relevant_ans > 0:
                
                answers_word.append(answer_word_group)
                answers_word_len.append(answer_word_group_len)
                labels.append(label_group)
                answers.append(answer_group)
                
                rawText                 = data_rows[i][QUESTION_INDEX].lower().replace("'", "")
                word_tokens             = re.sub("[^a-zA-Z0-9]", " ", rawText).split() 
                removing_stopwordsLists = [word for word in word_tokens if word not in stopwordsList]
                lemmatized_word         = [wordnet_lemmatizer.lemmatize(word) for word in removing_stopwordsLists]
                questions_word.append(lemmatized_word)
                questions_word_len.append(len(removing_stopwordsLists))
                questions.append(rawText)
                
            else:
                n_filtered_ans        += 1
                n_relevant_ans         = 0
            
    return answers_word, answers_word_len, answers, questions_word, questions_word_len, questions, labels