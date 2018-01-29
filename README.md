# Astroturfing-Review-System

![alt text](http://www.digitalstrategyconsulting.com/netimperative/news/fake%20reviews.jpg)

Created By : **Eashan Adhikarla**

Date       : **November 2015**

This is a filtering model with customized machine learning algorithm to filter-out the fake reviews and hence the fake reviewers from committing review frauds. I used amazon reviews data sets to built a stochastic model on top of it.

### Problem Statement:
Consumer reviews are now part of everyday decision-making. Yet, the credibility of these reviews is fundamentally undermined when businesses commit review fraud, creating fake reviews for themselves or their competitors. Investigation of the economic incentives to commit review fraud on the popular review platform Amazon and Yelp, using two complementary approaches and datasets. The economists Micheal Luca and Georgios Zervas begin by analyzing restaurant reviews that are identified by Yelp’s filtering algorithm as suspicious, or fake and treat these as a proxy for review fraud. They presented four main findings. First, roughly 16% of restaurant reviews on Yelp are filtered. These reviews tend to be more extreme (favorable or unfavorable) than other reviews, and the prevalence of suspicious reviews has grown significantly over time. Second, a restaurant is more likely to commit review fraud when its reputation is weak, i.e., when it has few reviews, or it has recently received bad reviews. Third, chain restaurants which benefit less from Yelp are also less likely to commit review fraud. Fourth, when restaurants face increased competition, they become more likely to receive unfavorable fake reviews. 

### Objective:
Using a separate dataset, I am trying to analyze reviewers that were caught soliciting fake reviews through a sting conducted by Amazon. 

### Prerequisites:

* 'os' for loading os folder paths
* 'pandas' for making dataframes
* 'numpy' for making arrays
* 'sklearn.metrics' for accuracy score, precision score, recall score, f1 score
* 'sklearn.cross_validation' for splitting the dataset
* 'CountVectorizer()' for extracting features from text in numerical form
* 'Multinomial Naive Bayes for importing naive bayes multinomial method classifier

### Approach:

a.) Import the datasets and store the data in three columns: **Polarity of the review | Review itself | True or Deceptive as _('t' or 'd')_**.

b.) Converting **'t'** to **1** and **'d'** to **0** because I will be using this as my target value and the review as my feature.

c.) Splitting the Review data into testing data and training data (0.3 and 0.7 respectively).

d.) Using **CountVectorizer()** to extract numeric features of each of the review as classifier can only use numeric data to compute something.

e.) Using Multinomial Naive Bias method classifier to classify the reviews as _Deceptive/True_.

### Datasets: 
* http://myleott.com/op_spam/
* https://www.kaggle.com/bittlingmayer/amazonreviews

### Licensing:

Unless otherwise stated, the source code and trained Python model files are copyright and licensed under the Apache 2.0 License(). Portions from the following third party sources have been modified and are included in this repository. 
