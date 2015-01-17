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


start =  datetime.now()

def evaluate_cross_validation(clf, X, y, K):
    # create a k-fold cross validation iterator of k=5 folds
    cv = KFold(len(y), K, shuffle=False, random_state=0)
    # by default the score used is the one returned by score method of the estimator (accuracy)
    scores = cross_val_score(clf, X, y, cv=cv)
    # iterate over the K folds
    k_train_scores = np.zeros(K)
    k_test_scores = np.zeros(K)
    for j, (train, test) in enumerate(cv):
            # fit the classifier in the corresponding fold
            # and obtain the corresponding accuracy scores on train and test sets
        clf.fit([X[k] for k in train], y[train])
        k_train_scores[j] = clf.score([X[k] for k in train], y[train])
        k_test_scores[j] = clf.score([X[k] for k in test], y[test])
    print scores
    print ("Mean score: {0:.3f} (+/-{1:.3f})").format(
        np.mean(scores), sem(scores))

def classifier_age(row):
         if row["Age"] > 20:
             return 1
         else:
             return 0

def classifier_age_var(row):

             return row['Age']*row['Age']


def classifier_fare(row):
         if row["Fare"] > 15:
             return row['Fare']*row['Fare']
         else:
             return 1
print(sklearn.__path__)
df = pd.read_csv('train.csv',header=0)

cols = ['Name','Ticket','Cabin']
df = df.drop(cols,axis=1)
df = df.dropna()

dummies = []
cols = ['Sex','Pclass','Embarked']
for col in cols:
    dummies.append(pd.get_dummies(df[col]))
titanic_dummies = pd.concat(dummies, axis=1)
df = pd.concat((df,titanic_dummies),axis=1)

df['Age'] = df['Age'].interpolate()
df["Age*"] = df.apply(classifier_age_var, axis=1)
df["Age**"] = df.apply(classifier_age, axis=1)
df["Fare*"] = df.apply(classifier_fare, axis=1)
df = df.drop(['Sex','Parch','SibSp','Pclass','Embarked','Age'],axis=1)
df = df.drop([3],axis=1)
#df['Age'] = df['Age'].var()
# df['Sex'].str.contains('female')["Sex"] = 1
#df.info()


#plotting data
#plt.plot(df['Age'].values)
#plt.plot(df['Survived'].values*40)
#plt.plot(df['female'].values*20)
#plt.show()
#exit()


X = df.values

y = df['Survived'].values
X = np.delete(X,1,axis=1)
#clf = ensemble.GradientBoostingClassifier(n_estimators=100)
#clf = tree.DecisionTreeClassifier(max_depth=10,random_state=None)
clf = ensemble.RandomForestClassifier(n_estimators=200)
evaluate_cross_validation(clf,X,y,8)
#exit()
#X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2,random_state=0)
print("training the classifier....")

#print clf
#clf.fit (X_train, y_train)
#accuracy=clf.score (X_test, y_test)

#print("the acuracy is....") + str(accuracy)

df = pd.read_csv('test.csv',header=0)

cols = ['Name','Ticket','Cabin']
df = df.drop(cols,axis=1)
df['PassengerId'] = df['PassengerId'].interpolate()
df['SibSp'] = df['SibSp'].interpolate()
df['Parch'] = df['Parch'].interpolate()
#df['Fare'] = df['Fare'].interpolate()


dummies = []
cols = ['Sex','Pclass','Embarked']
for col in cols:
    dummies.append(pd.get_dummies(df[col]))
titanic_dummies = pd.concat(dummies, axis=1)
df = pd.concat((df,titanic_dummies),axis=1)

df['Age'] = df['Age'].interpolate()
df['Fare'] = df['Fare'].interpolate()
df["Age*"] = df.apply(classifier_age_var, axis=1)
df["Age**"] = df.apply(classifier_age, axis=1)
df["Fare*"] = df.apply(classifier_fare, axis=1)
df = df.drop(['Sex','Parch','SibSp','Pclass','Embarked',3,'Age'],axis=1)
print(df.head())
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
#st.to_csv("submission_%s" % str(clf.best_estimator_).split("(")[0],index=False)