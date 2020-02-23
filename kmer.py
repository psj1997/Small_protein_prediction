"""
using 3mer to build a easy baseline model of the small protein prediction
"""
import numpy as np
import pandas as pd

coden_list = ['AAA', 'AAT', 'AAC', 'AAG', 'ATA', 'ATT', 'ATC', 'ATG',
            'ACA', 'ACT', 'ACC', 'ACG', 'AGA', 'AGT', 'AGC', 'AGG',
            'TAA', 'TAT', 'TAC', 'TAG', 'TTA', 'TTT', 'TTC', 'TTG',
            'TCA', 'TCT', 'TCC', 'TCG', 'TGA', 'TGT', 'TGC', 'TGG',
            'CAA', 'CAT', 'CAC', 'CAG', 'CTA', 'CTT', 'CTC', 'CTG',
            'CCA', 'CCT', 'CCC', 'CCG', 'CGA', 'CGT', 'CGC', 'CGG',
            'GAA', 'GAT', 'GAC', 'GAG', 'GTA', 'GTT', 'GTC', 'GTG',
            'GCA', 'GCT', 'GCC', 'GCG', 'GGA', 'GGT', 'GGC', 'GGG']
coden_dict = {}
count = 0
for coden in coden_list:
    count += 1
    coden_dict[coden] = count

#print(coden_dict)
def seq_to_vector(seq):
    vector = np.zeros((1,64))
    for i in range(len(seq)-3):
        #print(seq[i:i+3])
        if seq[i:i+3] in coden_dict:
            vector[0][coden_dict[seq[i:i+3]]-1] += 1
    return vector

training_data = pd.read_csv('/home1/pansj/Small_protein_prediction/training_data_whole.csv',index_col=0).values
data = training_data[:,0]

label = training_data[:,1]
label = label.reshape(len(label),-1)
label[label == -1] = 0
label = np.array([y[0] for y in label]).reshape(len(label),1)

data_vector = []

for seq in data:
    seq_vector = seq_to_vector(seq)
    data_vector.append(seq_vector)

train_data = np.array(data_vector)
train_data = train_data.reshape(len(train_data),64)
print(train_data.shape)
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
skf = StratifiedKFold(n_splits=5, shuffle=True)

accuracy = []
recall = []
for train_index, test_index in skf.split(train_data, label):
    X_train, Y_train = train_data[train_index], label[train_index]
    X_test, Y_test = train_data[test_index], label[test_index]

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    clf = LogisticRegression()
    clf.fit(X_train,Y_train)
    acc = clf.score(X_test,Y_test)
    pre = clf.predict(X_test)
    num = 0
    for i in range(len(pre)):
        if pre[i] == Y_test[i] and pre[i] == 1:
            num += 1
    # accuracy.append(score)
    accuracy.append(acc)
    recall.append(num/len(X_test))

print(accuracy)
print('acc:',np.average(accuracy))
print(recall)
print('recall:',np.average(recall))




