# Astroturfing-Review-System

![alt text](http://www.digitalstrategyconsulting.com/netimperative/news/fake%20reviews.jpg)

**Created By : Eashan Adhikarla
| Date       : November 2015**

This is a filtering model with customized machine learning algorithm to filter-out the fake reviews and hence the fake reviewers from committing review frauds. I used amazon reviews data sets to built a stochastic model on top of it.

## Problem Statement:
Consumer reviews are now part of everyday decision-making. Yet, the credibility of these reviews is fundamentally undermined when businesses commit review fraud, creating fake reviews for themselves or their competitors. Investigation of the economic incentives to commit review fraud on the popular review platform Amazon and Yelp, using two complementary approaches and datasets. The economists Micheal Luca and Georgios Zervas begin by analyzing restaurant reviews that are identified by Yelp’s filtering algorithm as suspicious, or fake and treat these as a proxy for review fraud. They presented four main findings. First, roughly 16% of restaurant reviews on Yelp are filtered. These reviews tend to be more extreme (favorable or unfavorable) than other reviews, and the prevalence of suspicious reviews has grown significantly over time. Second, a restaurant is more likely to commit review fraud when its reputation is weak, i.e., when it has few reviews, or it has recently received bad reviews. Third, chain restaurants which benefit less from Yelp are also less likely to commit review fraud. Fourth, when restaurants face increased competition, they become more likely to receive unfavorable fake reviews. 

## Objective:
Using a separate dataset, I am trying to analyze reviewers that were caught soliciting fake reviews through a sting conducted by Amazon. 

## Prerequisites:

* nltk
* sklearn and scipy
* pandas and numpy

## Approaches Used :

### Method 1 : 

* Sentiment Analysis
* Coontent Similarity
* Latent Symantic Analysis

### Sentimental Analysis

Sentimental Analysis is contextual mining of text which identifies and extracts subjective information in source material and helping a business to understand the social sentiment of their brand, product or service while monitoring online conversations.

```
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=2000,min_df =3 ,max_df = 0.6, stop_words = stopwords.words("english"))    
X = vectorizer.fit_transform(corpus).toarray()

#Cretaing TF-IDF from BOW
from sklearn.feature_extraction.text import TfidfTransformer

transformer = TfidfTransformer()
X = transformer.fit_transform(X).toarray()

#Spliting for testing and training

from sklearn.model_selection import train_test_split

text_train,text_test,sent_train,sent_test = train_test_split(X,y,test_size=0,random_state=0)
\# here text size = 0 , so that all the data will be used for the training purpose only

\# Training our classifier
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(text_train,sent_train)
```

### Latent Symantic Analysis

Latent semantic analysis (LSA) is a technique in natural language processing, in particular distributional semantics, of analyzing relationships between a set of documents and the terms they contain by producing a set of concepts related to the documents and terms. A matrix containing word counts per paragraph (rows represent unique words and columns represent each paragraph) is constructed from a large piece of text and a mathematical technique called singular value decomposition (SVD) is used to reduce the number of rows while preserving the similarity structure among columns Values close to 1 represent very similar words while values close to 0 represent very dissimilar words.

```
# Latent symantic analysis
# it will analyse all reviews and determine all reviews belong to the same concept
def LSA(text):
    #text is list of reviews of same product
    
    # Created TF-IDF Model
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(text)
    
    # Created SVD(Singular Value Decomposition)
    lsa = TruncatedSVD(n_components = 1,n_iter = 100)
    lsa.fit(X)
    
    terms = vectorizer.get_feature_names()
    concept_words={}
    for j,comp in enumerate(lsa.components_):
        componentTerms = zip(terms,comp)
        sortedTerms = sorted(componentTerms,key=lambda x:x[1],reverse=True)
        sortedTerms = sortedTerms[:10]
        concept_words[str(j)] = sortedTerms
     
    sentence_scores = []
    for key in concept_words.keys():
        for sentence in text:
            words = nltk.word_tokenize(sentence)
            scores = 0
            for word in words:
                for word_with_scores in concept_words[key]:
                    if word == word_with_scores[0]:
                        scores += word_with_scores[1]
            sentence_scores.append(scores)
    return sentence_scores
```
### Content Similarity

With cosine similarity, we need to convert sentences into vectors.Difference in the angle of these determines the similarity between two reviews.

```
from nltk.corpus import stopwords from sklearn.feature_extraction.text import TfidfVectorizer from sklearn.metrics.pairwise import cosine_similarity

tfidf_vectorizer = TfidfVectorizer()
for i in range(len(dataset)):
    
    reviews = [str(dataset["review_body"][i])]
    
    tfidf_vectorizer.fit_transform(reviews)

tfidf_matrix = tfidf_vectorizer.fit_transform(reviews)
    
    #creates TF-IDF Model
    tfidf_list = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix).tolist()
    # Creates matrix based on document similarity
         
    # To check similarity b/w 2 reviews 
    for k in range(1,len(tfidf_list[0])):
                
        if(tfidf_list[0][k]>0.6):
            # 0.6 is defind for the simmilarity level
            
            remove_reviews.append(dataset["review_id"][i+k])
            # i+k is to get the review id of the review
```

### Method 2 : 

1. Import the datasets and store the data in three columns: **Polarity of the review | Review itself | True or Deceptive as _('t' or 'd')_**.

2. Converting **'t'** to **1** and **'d'** to **0** because I will be using this as my target value and the review as my feature.

3. Splitting the Review data into testing data and training data (0.3 and 0.7 respectively).

4. Using **CountVectorizer()** to extract numeric features of each of the review as classifier can only use numeric data to compute something.

5. Using Multinomial Naive Bias method classifier to classify the reviews as _Deceptive/True_.

## Datasets: 

* Amazon - https://snap.stanford.edu/data/web-Amazon.html (Stanford University)
* Amazon - https://www.kaggle.com/bittlingmayer/amazonreviews
* Yelp   - http://myleott.com/op_spam/

## Licensing:

Unless otherwise stated, the source code and trained Python model files are copyright and licensed under the [MIT License](https://github.com/eashanadhikarla/Astroturfing-Review-System/blob/master/LICENSE). Portions from the following third party sources have been modified and are included in this repository.
