
#Importing required library
import pandas as pd
import numpy as np
from collections import Counter

def KNN_1sample(target, X, y, K):
	
	dists = np.linalg.norm(X-target, ord=2, axis = 1)
	sorted_idxs = np.argsort(dists)
	label_cnt = np.bincount(y[sorted_idxs[:K]])
	return np.argmax(label_cnt)
	
	
def KNN_np(Xt, X, y, K):
	
	return np.apply_along_axis(KNN_1sample, 1, Xt, *(X, y, K))


#headernames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

#opening the csv file
dataset = pd.read_csv("data/IRIS.csv") # 不要制定head 或者设成None 否则第一行会被当成data 从而dtype为object
# df= pd.read_csv("data/IRIS.csv", header=[0])
dataset.head()



#Seperating the input features and output labels
# X = dataset.iloc[:, :-1].values #iloc X contains columns
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
#converting text labels to numeric form
labels, unique = pd.factorize(y) ##TODO ??pd.factorize


#splitting data in test and train segments
from sklearn.model_selection import train_test_split
import pandas as pd


res = KNN_1sample(X[5], X, labels, 5)
res2 = KNN_np(X[[5,25,50,100]], X, labels, 10)
print("Dont")

def d2numeric(dataset):
    return dataset[dataset.columns[:-1]][1:].apply(pd.to_numeric, axis=0)


def dist(target, X):
    return np.apply_along_axis(np.linalg.norm, 1, X-target, **{"ord":2, "axis":1})

def dist_pd(target, X):
    return (X-target).apply(np.linalg.norm, axis=1)
    cols = X.columns
    df = X[cols] - target[cols]
    return df.apply(lambda values: sum([v**2 for v in values]), axis=1)
    return df.apply(lambda x: (x**2).sum()**.5, axis=1)
    return np.linalg.norm(df[['X','Y','Z']].values,axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size = 0.40)
X_train = X_train.astype('float')
X_test = X_test.astype('float')
#X_train, X_test = d2numeric(X_train), d2numeric(X_test)

dist(X_test[1], X_train)
print("Done")