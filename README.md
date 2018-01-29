# Astroturfing-Review-System

Created By : Eashan Adhikarla
Date       : 4th November'2015

This is a filtering model with machine learning algorithm to filter-out the fake reviews and hence the fake reviewers from committing review frauds. I used amazon reviews data sets to built a stochastic model on top of it.

# Prerequisites:

* 'os' for loading os folder paths
* 'pandas' for making dataframes
* 'numpy' for making arrays
* 'sklearn.metrics' for accuracy score, precision score, recall score, f1 score
* 'sklearn.cross_validation' for splitting the dataset
* 'CountVectorizer()' for extracting features from text in numerical form
* 'Multinomial Naive Bayes for importing naive bayes multinomial method classifier

Step 1: I am importing the datasets and storing the datas in three columns: 
* Polarity of the review
* Review itself
* True or Deceptive as ('t' or 'd')

Step 2: Converting 't' to 1 and 'd' to 0 because I will be using this as my target value and the review as my feature.

Step 3: Splitting the Review data into testing data and training data (0.3 and 0.7 respectively).

Step 4: Using CountVectorizer() to extract numeric features of each of the review as classifier can only use numeric data to compute something.

Step 5: Using Multinomial Naive Bias method classifier to classify the reviews as Deceptive/True.

Datasets: 
* http://myleott.com/op_spam/
* https://www.kaggle.com/bittlingmayer/amazonreviews
