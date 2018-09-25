from preprocessing import X_train, y_train
from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn.cross_validation import cross_val_score
import numpy as np

## 多项式朴素贝叶斯

model_NB = MNB()
model_NB.fit(X_train, y_train)

print(cross_val_score(model_NB, X_train, y_train, cv = 20, scoring = 'roc_auc'))
print("MultinomialNB classifier cross_validation score:", np.mean(cross_val_score(model_NB, X_train, y_train, cv = 20, scoring = 'roc_auc')))

from sklearn.linear_model import LogisticRegression as LR
from sklearn.grid_search import GridSearchCV

## 逻辑回归

# 设定grid search的参数
grid_values = {'C' : [30]}

# 设定打分为roc_auc
model_LR = GridSearchCV(estimator = LR(C=1.0, class_weight=None, dual=True, fit_intercept=True, intercept_scaling=1, penalty='l2', random_state=0, tol=0.0001), param_grid = grid_values, scoring = 'roc_auc', cv = 20, fit_params={}, iid=True, n_jobs=1, refit=True, verbose=0)

model_LR.fit(X_train, y_train)

print("LogisticRegression gird scores:", model_LR.grid_scores_)