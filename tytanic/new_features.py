__author__ = 'anirudha'
import pandas as pd
import numpy as np
from sklearn import cross_validation
import sklearn
from sklearn import tree
from sklearn import ensemble
from datetime import datetime
from matplotlib import pyplot as plt
from sklearn.cross_validation import cross_val_score, KFold
from scipy.stats import sem

def evaluate_cross_validation(clf, X, y, K):
    print("training the classifier....")
    # create a k-fold cross validation iterator of k=5 folds
    cv = KFold(len(y), K, shuffle=False, random_state=0)
    # by default the score used is the one returned by score method of the estimator (accuracy)
    scores = cross_val_score(clf, X, y, cv=cv)
    # iterate over the K folds
    k_train_scores = np.zeros(K)
    k_test_scores = np.zeros(K)
    for j, (train, test) in enumerate(cv):
            # fit the classifier in the corresponding fold
            # and obtain the corresponding accuracy scores on train and test sets4
        clf.fit([X[k] for k in train], y[train])
        k_train_scores[j] = clf.score([X[k] for k in train], y[train])
        k_test_scores[j] = clf.score([X[k] for k in test], y[test])
    print scores
    print ("Mean score: {0:.3f} (+/-{1:.3f})").format(
        np.mean(scores), sem(scores))
    return clf

df = pd.read_csv('titanic_train',header=0)
X = df.values
print X.shape
y = df['Survived'].values
X = np.delete(X,1,axis=1)
#clf = ensemble.GradientBoostingClassifier(n_estimators=100)
#clf = tree.DecisionTreeClassifier(max_depth=10,random_state=None)
clf = ensemble.RandomForestClassifier(n_estimators=1000)
clf.fit(X,y)
#clf = evaluate_cross_validation(clf,X,y,4)
#X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.5,random_state=0)
#clf.fit (X_train, y_train)
#accuracy=clf.score (X_test, y_test)
#print("the acuracy is....") + str(accuracy)

df = pd.read_csv('titanic_test',header=0)
X_results = df.values
print X_results.shape
y_results = clf.predict(X_results)
output = np.column_stack((X_results[:,0],y_results))
#print output
df_results = pd.DataFrame(output.astype('int'),columns=['PassengerID','Survived'])
print(df_results.shape)
#print(df_results.head())
#df_results.to_

# csv('titanic_results_%s'% str(start),index=False)
df_results.to_csv('titanic_results',index=False)