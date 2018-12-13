# Alternus Vera Project

Course code : **CMPE-257**

Group name : **Codebusters**

Name: **Harini Balakrishnan (010830755)**

-----

GitHub URL: https://github.com/HariniGB/AlternusVera


### Liar Liar Pants on Fire Dataset Description
- It has 3 files test, training and valid.
- Each file has 14 columns

    Column 1: the ID of the statement ([ID].json).

    Column 2: the label.

    Column 3: the statement.

    Column 4: the subject(s).

    Column 5: the speaker.

    Column 6: the speaker's job title.

    Column 7: the state info.

    Column 8: the party affiliation.

    Column 9-13: the total credit history count, including the current statement.

    Column 14: the context (venue / location of the speech or statement).

### Process of My Approach
- Load the Data
- Distillation Process
    - Data Cleaning and Text Preprocessing
    - Visualization
    - **Feature 1 :** Sentiment Analysis using Vader
    - **Feature 2 :** LDA Topic Modeling
    - **Feature 3 :** Sensationalism Analysis Cosine Similarity
- Counter Vectorization and classification models
- TF-IDF Vectorization  and classification models
- Word2Vec word embedding and TSNE Visualization
- Doc2Vec Tagging and classification models
- Compare Counter vs TF-IDF vs Doc2Vec
- Vector Classification Modeling
- Ranking and Importance


**Classification algorithms:**
    - Naives Bayes regression
    - Logistic regression
    - SVM Stochastic Gradient Descent
    - Linear SVM Classifier
    - RandomForestClassifier


### Feature Selection
**Top Features Selected based on research articles**

1. Political Affiliation
2. Sensationalism
3. Click bait
4. Context Modeling
5. Spam

*Other simple features as a part of distillation:*
6. Sentiment Analysis
7. LDA Topic Modeling
8. Ranking


### Team Contributions:

|Features  |  Member |
|-----|-----|
| Sensationalism                         |  Harini Balakrishnan|
| Political Affiliation                   |  Anushri Srinath Aithal|
| Context Modeling                       |  Sunder Thyagarajan |
| Clickbait                              |  Ravi   Katta|
| Spam                                   |  All Memebers|


### My Contributions:

| Versions | Features | DataEnrichment & Corpus | Explanation |
|-----|-----|-----|-----|
| **Part 1**|  *Sentiment Analysis* | Vader Sentiment Intensity Analyses, SenticNet5 | Tokenization, Normalization, Stemming,  CounterVectorization, TF-IDF Vectorization, Doc2Vec and classification models  |
| **Part 2**|  *Distillation* |    GoogleNews-vectors-negative300 |    Stop words, Lemmentization, Spell Check, LDA Topic Modeling, word2vec, lda2vec |
| **Part 3**|  *Sensationalism* | The Persuasion Revolution Website | Doc2Vec, word2vec, TF-IDF Vectorization, Cosine Similarity, Compared three vectorization models|

#### Enrichment

- SenticNet5 for sentiment and sensationalism corpus
- Google News 3million words corpus for spell check

#### Libraries

- NLTK
- Gensim
- Numpy
- Pandas
- CSV
- WordCloud
- Seaborn
- Scipy
- Regualr Expression
- Matplotlib
- Sklearn


#### What did I try and What worked?

> Initially I preprocessed the given dataset using NLTK in-build libraries for tokenization, stopwords removal, stemming and lemmentization. Then I decided to visualize the cleaned data using WordCloud. I decided to extract compound features like Sentiment, Sensationalism and LDA Topic score and utilized it to classify the news document as fake or not. I tried three methods.First I tired CountVectorizer with which I got 95% for sentiment and 63.45% for sensationalism. Next I tried TfidfVectorizer which gave me 97% accuracy for sentiment and  61% for sensationalism. Finally I tried Doc2Vec which gave 50% for sensationalism and 91% accuracy for sentiment feature.

#### What did not work?

> As I was taking the sentiment intensity of each word and aggregating the words to get single vector for each document, I tired to get a vector for each word. I tried Word2Vec Alternus Vera Paper (Sentiment analysis) - Draft 1 library. There are two types of architecture options in Word2Vec: skip-gram (default) and CBOW (continuous bag of words). Most of time, skip-gram is little bit slower but has
more accuracy than CBOW. CBOW is the method to predict one word by whole text; therefore, small set of data is more favorable. On the other hand, skip-gram is totally
opposite to CBOW. With the target word, skip-gram is the method to predict the words around the target words. The more data we have, the better it performs. As the architecture, there are two training algorithms for Word2Vec: Hierarchical softmax (default) and negative sampling. I used the default. The Word2Vec provides vectorization for each word. Which causes dimensional issues with the 'senti_word_vector' column of each document. I performed Vector Averaging which gave me single vector for each document but with different length of the vector.


#### What alternatives did you try?

> I couldn’t perform classification model as the dimension of the vector varies for each document. My solution is to try Doc2Vec which will provide a vector of fixed size for the entire document instead of each word in the entire corpus. In the word2vec architecture, the two algorithm names are “continuous bag of words” (CBOW) and “skip-gram” (SG); in the doc2vec architecture, the corresponding algorithms are “distributed memory” (DM) and “distributed bag of words” (DBOW).I performed Doc2Vec using all the three features to predict if a document is fake or not and obtained 56% accuracy.

----

## RESULT

### Compare CountVectorizer vs TF-IDF vs Doc2Vec for Sentiment Analysis


1. CountVectorizer


| Models  | Accuracy |
|---|---|
| Naive Bayes | 92.1% |
| Logistic Regression | 94.8% |
| Linear SVM classifier | 95% |
| Stochastic Gradient Descent | 94.1% |
| Randome Forest Classifier | 92.0% |

2. TF-IDF Vectorizer


|Models  | Accuracy |
|---|---|
| Naive Bayes | 95.8% |
| Logistic Regression | 95.8% |
| Linear SVM classifier | 96.7% |
| Stochastic Gradient Descent | 95.8% |
| Randome Forest Classifier | 96.1% |

3. Doc2Vec


|Models  | Accuracy |
|---|---|
| Logistic Regression | 91.1% |
| Linear SVM classifier | 91.1% |
| Stochastic Gradient Descent | 91.1% |
| Randome Forest Classifier | 83.9% |


---

### Compare CountVectorizer vs TF-IDF vs Doc2Vec for Sensationalism

1. CountVectorizer


| Models  | Accuracy |
|---|---|
| Naive Bayes | 59.9% |
| Logistic Regression | 63% |
| Linear SVM classifier |63.45% |
| Stochastic Gradient Descent | 63.4% |
| Randome Forest Classifier | 61% |

2. TF-IDF Vectorizer


|Models  | Accuracy |
|---|---|
| Naive Bayes | 56.4% |
| Logistic Regression | 60 % |
| Linear SVM classifier | 61% |
| Stochastic Gradient Descent | 48 % |
| Randome Forest Classifier | 57 % |

3. Doc2Vec


|Models  | Accuracy |
|---|---|
| Logistic Regression | 49.3 % |
| Linear SVM classifier | 49.4% |
| Stochastic Gradient Descent | 48.4 % |
| Randome Forest Classifier | 49.8 % |


##### Inferences

The Count vectorization Linear SVM outnumbered both the TF-IDF and Doc2Vec because CountVectorization performed binary vectorization of words. Whereas TF-IDF takes probabilistic approach and gives more accurate score for each word. Doc2Vec wasn't rich or big enough for the actual news because of the limited content which wasn't enough for the model to understand to generate sensible embedding.

As my team we decided to classify the final vectorization using Doc2Vec. With 56% accuracy on sensationalism, we decided to provide **0.15** scalar weight for this feature in the polynomial equation.


