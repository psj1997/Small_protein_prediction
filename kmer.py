"""
using 3mer to build a easy baseline model of the small protein prediction
"""
import numpy as np
import pandas as pd
from itertools import product
from sklearn.model_selection import train_test_split, StratifiedKFold,GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score

codon_list = [''.join(cs) for cs in product('ATCG', 'ATCG', 'ATCG','ATCG','ATCG')]
codon2ix = {c:i for i,c in enumerate(codon_list)}

def seq_to_vector(seq,k):
    vector = np.zeros(4**k)
    for i in range(len(seq)-k):
        #print(seq[i:i+3])
        ix = codon2ix.get(seq[i:i+k])
        if ix is not None:
            vector[ix] += 1
    return vector

def Cross_validation(is_group = False):
    training_data = pd.read_csv('/home1/pansj/Small_protein_prediction/training_data_cluster.csv',index_col=0).values
    print(training_data.shape)
    data = training_data[:, 0]
    label = training_data[:, 1]
    cluster = training_data[:, 3]
    print('cluster:', len(set(cluster)))
    label = label.reshape(len(label), -1)
    print(np.sum(label > 0), len(label))
    label[label == -1] = 0
    label = np.array([y[0] for y in label]).reshape(len(label), 1)

    train_data = np.array([seq_to_vector(seq, 5) for seq in data])
    print(len(train_data))

    accuracy = []
    recall = []
    precision = []
    f1 = []

    if is_group == False:
        print('CV:')
        scaler = StandardScaler()
        skf = StratifiedKFold(n_splits=5, shuffle=True)
        for train_index, test_index in skf.split(train_data, label):
            print(len(train_index),len(test_index))
            X_train, Y_train = train_data[train_index], label[train_index]
            X_test, Y_test = train_data[test_index], label[test_index]
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.fit_transform(X_test)
            clf = RandomForestClassifier()
            #clf = LogisticRegression()
            clf.fit(X_train,Y_train)
            acc = clf.score(X_test,Y_test)
            pre = clf.predict(X_test)
            accuracy.append(acc)
            recall.append(recall_score(Y_test,pre))
            precision.append(precision_score(Y_test,pre))
            f1.append(f1_score(Y_test,pre))
    else:
        print('Group_cv')
        scaler = StandardScaler()
        skf = GroupKFold(n_splits=5)
        for train_index, test_index in skf.split(train_data, label, cluster):
            print(len(train_index), len(test_index))
            X_train, Y_train = train_data[train_index], label[train_index]
            X_test, Y_test = train_data[test_index], label[test_index]

            X_train = scaler.fit_transform(X_train)
            X_test = scaler.fit_transform(X_test)

            clf = RandomForestClassifier()
            # clf = LogisticRegression()
            clf.fit(X_train, Y_train)
            acc = clf.score(X_test, Y_test)
            pre = clf.predict(X_test)

            accuracy.append(acc)
            recall.append(recall_score(Y_test, pre))
            precision.append(precision_score(Y_test, pre))
            f1.append(f1_score(Y_test, pre))

    print(accuracy)
    print('acc:',np.average(accuracy))
    print(recall)
    print('recall:',np.average(recall))
    print(precision)
    print('precision:',np.average(precision))
    print(f1)
    print('f1:',np.average(f1))


if __name__ == '__main__':
    Cross_validation(is_group=True)




