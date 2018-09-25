# naive bayes and logistic regression for sentiment analysis

## Brief introduction

train a naive bayes model and logistic regression model to predict the sentiment according to movie comments, and compare the scores of them.

all.zip includes the training data and test data, and the submission format.

preprocessing.py includes data preprocessing of training data and test data, and initialize the TFIV object.

model_building.py includes constructing a baseline model of naive bayes and logistic regression.

test_save.py uses the models trained before to predict the label of the test data and save it as the form of .csv .

## late submission
we compare the scores of two models in late submission, the Bayes is lower than logistic regression, which is 85.746 about rank of 300th, on the contrast, score of logistic regression is 0.88948 which is about rank of 260th.

However, if happens to be a huge dataset, the time cost of training a logistic regression is much longer than training a bayes.