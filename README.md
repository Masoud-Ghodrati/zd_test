# ZD Challenge Overview

The provided [dataset](zendesk_challenge.tsv)  contains question and answer pairs. Every question in the dataset can have multiple relevant (correct) and irrelevant (incorrect) answers. The data includes a column with labels referring to relevant (correct=1) and irrelevant (incorrect=0) answers.

As the aim (of the challenge) is rather open, I took few approaches (simple to more complex) with the aim to see if I can select the relevant answer(s) using the information in the questions and answers sets (let's say the questions are customers' requests/reviews/opinions about a product that has information document for customers service). 
In summary I tried these approaches in the time that I had: 

1. Word matching analysis(see this code: [`ZenِDesk_BasicAnalysis.py`](ZenِDesk_BasicAnalysis.py)): it basically counts the number of matching non-stopwords in the questions and provided answers to find their relevance. 
2. Sentence root matching (see this code: [`ZenDesk_SentRootMatch.py`](ZenDesk_SentRootMatch.py)): it uses root parsing methods to get the roots for every question and corresponding answers to find the most relevant answer.
3. Sentence embeddings methods (see this code: [`ZenDesk_InferSent.py`](ZenDesk_InferSent.py)): it uses sentence embedding methods to encode questions and answers to numeric vectors and measures the similarity between Qs and As vectors to find the relevance.

The performance of the methods was evaluated using F1-score (F1 of 1 is perfect and 0 is bad). Higher F1 scores show better performance of the model (e.g., meaning that the model is successful in providing relevant information to customers' requests).

|  Methods: |  1  |  2  |  3  |
|-----------|-----|-----|-----|
| F1-score  |0.36 |0.31 | 0.34|


Note that F1-scores at this stage are not very high but there is room for improvement by better pre-processings and models.
The following shows implementation details.

## Other TO DO things:

From the technical side, if I had more time, I could play with other models (e.g., [Google BERT](https://github.com/google-research/bert)) and large datasets (e.g., [Natural Questions](https://ai.google.com/research/NaturalQuestions)) to see how they perform on the provided data. I also did not try to develop a supervised model based on the provided data to evaluate its performance. 

Other more things that I could play with if I had more time are:
* Categorize relevant answers and questions to a number of groups based on their content to see what are the several main contents. This then would help to, for example, provide related information on the company website or social media based on the content (and customers interests)

* Grouping the questions into several main contents also helps to see how many times customers had similar requests from, let’s say, customer service (such things can also be used in the targetted advertisement)

* Handling and processing answers in a better and even combining the relevant one to a new one can also be helpful to generate more comprehensive answers to a question with similar content. This would help to provide clearer information to customers with similar requests


# Implementation Details
## Word matching analysis
After some data cleaning and pre-processing, I did some simple text analysess. This code [`ZenِDesk_BasicAnalysis.py`](ZenِDesk_BasicAnalysis.py) will do the job. You just need to change these paths:

```python
# data path and file name
data_path   = '.../...'
result_path = '.../...'
```

Before doing other stuff, the first that comes to mind is doing some word frequency analysis to see if the content of questions and answers provides us with any insight about what the whole text is about. For example, what sort of topics/words costumers cover in their requests/questions and what sort of information is provided to them. This helps the production line to maybe update the focus to topics of customers interest.

An easy way for this is by using visualisation methods such as [Word Cloud](http://amueller.github.io/word_cloud/) as it makes things simpler to grasp at a glance.
Results that the text (Qs and As) is mostly about a number of topics such as *united states*, *war*, and *country* but it also covers a range of other topics.


![Fig. 1](Word_Frequency.png)
Visualisation of word frequency in questions and answers



The 10 top most frequent words in questions and answers are: 

|  Q word  |   count  |  A word  |   count  |
|----------|----------|----------|----------|
|  state   |    26    |  state   |   956    |
|   made   |    26    |  united  |   592    |
|   mean   |    20    |   also   |   553    |
| country  |    20    |   one    |   481    |
|   war    |    19    |  world   |   417    |
|   name   |    19    |  first   |   409    |
|  county  |    19    |   war    |   402    |
|  first   |    17    | american |   384    |
|   used   |    15    |   year   |   379    |


Then, I took one step further and consider a simple word matching method. So, I counted the number of non-stopwords in the question that also occur in the answer sentence. The higher this number is between two pairs, then there is a higher chance that the answer is relevant to the question. The *word* can be rather useful in some simple applications (let's say a bot that provides information to customers about a particular hair product).

I also generated random labels for the answers to check if F1 score of *word matching method* is higher than random labels. This random process was done 1000 times. I used two randomization approach 1) randomized the indexes of the whole label column 2) randomized the indexes of each answer group labels. 

Results show that although the F1 score for *word matching method* is not very high, it is greater than a random answer guessing.

![Fig. 2](Word_Matching_Performance.png)
The performance of *word matching methods* compared to random distributions 


## Sentence root matching
Another way to approach this question is using (syntactic) dependency parsing. I used [NLTK](https://www.nltk.org/), and [SpaCy’s](https://spacy.io/) root parsing to get the roots for every question and corresponding answer. The goal is to see if the root of the question matches with all the roots/sub-roots of the answer. If there is a root matching between two pairs, then there is a higher chance that the answer is relevant to the question. I used F1 score for performance evaluation.

The code to run this part is [`ZenDesk_SentRootMatch.py`](ZenDesk_SentRootMatch.py). You only need to change the paths

```python
# data path and file name
data_path   = '.../...'
result_path = '.../...'
```

*F1 score based on sentnece root matching (between Qs and As)*: **0.3119**


## Sentence embedding methods
Sentence embedding methods encode words or sentences into fixed length numeric vectors which are pre-trained on a large text corpus. This can be very helpful as such text datasets cover a wide range of topics that can be very handy in applications similar to the ZD provided dataset.

The embeddings can then be used for various tasks like finding similarity between two sentences (in our case similarity between questions and answers).
Here, I used [InferSent](https://github.com/facebookresearch/InferSent) which is a sentence embeddings method that provides semantic representations for English sentences. It is trained on natural language inference data and generalizes well to many different tasks. The output of the model is a `numpy` array with a vector of dimension 4096 (for a sentence/word). Using these arrays, I found the similarities between questions and answers and predicted the labels based on the similarity.

The code to run this part is [`ZenDesk_InferSent.py`](ZenDesk_InferSent.py). Note that you need to first install [InferSent](https://github.com/facebookresearch/InferSent) to be able to run the code.

*F1 score based on Sentence embedding (between Qs and As)*: **0.34**
 
