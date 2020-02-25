"""
using 3mer to build a easy baseline model of the small protein prediction
"""
import numpy as np
import pandas as pd
from itertools import product
codon_list = [''.join(cs) for cs in product('ATCG', 'ATCG', 'ATCG')]
codon2ix = {c:i for i,c in enumerate(codon_list)}

def seq_to_vector(seq):
    vector = np.zeros(64)
    for i in range(len(seq)-3):
        #print(seq[i:i+3])
        ix = codon2ix.get(seq[i:i+3])
        if ix is not None:
            vector[ix] += 1
    return vector

training_data = pd.read_csv('data/training_data_whole.csv',index_col=0).values
data = training_data[:,0]

label = training_data[:,1]
label = label.reshape(len(label),-1)
label[label == -1] = 0
label = np.array([y[0] for y in label]).reshape(len(label),1)

train_data = np.array([seq_to_vector(seq) for seq in data])

from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
skf = StratifiedKFold(n_splits=5, shuffle=True)
from collections import Counter
accuracy = []
recall = []
precision = []
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score,precision_score
for train_index, test_index in skf.split(train_data, label):
    X_train, Y_train = train_data[train_index], label[train_index]
    X_test, Y_test = train_data[test_index], label[test_index]

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    #clf = RandomForestClassifier()
    clf = LogisticRegression()
    clf.fit(X_train,Y_train)
    acc = clf.score(X_test,Y_test)
    pre = clf.predict(X_test)


    accuracy.append(acc)
    recall.append(recall_score(Y_test,pre))
    precision.append(precision_score((Y_test,pre)))

print(accuracy)
print('acc:',np.average(accuracy))
print(recall)
print('recall:',np.average(recall))
print(precision)
print('precision:',np.average(precision))




