from keras.models import Sequential, load_model, model_from_config

MAX_SEQ_LENGTH = 153

def sequence_to_onehot(seq):
    """
    return the one_hot of the sequence 4*m A(1,0,0,0) T(0,1,0,0) G(0,0,1,0) C(0,0,0,1)
    :param seq:
    :param max_length:
    :return:
    """
    seq_one_hot = []
    for i in range(MAX_SEQ_LENGTH):
        if i < len(seq):
            if seq[i] == 'A':
                seq_one_hot.append([1,0,0,0])
            elif seq[i] == 'T':
                seq_one_hot.append([0,1,0,0])
            elif seq[i] == 'G':
                seq_one_hot.append([0,0,1,0])
            elif seq[i] == 'C':
                seq_one_hot.append([0,0,0,1])
            else:
                seq_one_hot.append([0,0,0,0])
        else:
            seq_one_hot.append([0,0,0,0])
    seq_one_hot = np.array(seq_one_hot).reshape(153,4)
    return seq_one_hot

from keras.layers import Conv1D,MaxPool2D,MaxPool1D,Flatten,Dense

def CNN_model():
    model = Sequential()
    model.add(Conv1D(filters=1,kernel_size=3,strides=1,activation='relu',padding='valid', data_format='channels_last',input_shape=(153,4)))
    model.add(MaxPool1D(pool_size=5,strides=1,padding='valid'))
    model.add(Flatten())
    model.add(Dense(64,activation='relu'))
    model.add(Dense(2,activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.summary()
    return model

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split,StratifiedKFold
import json

if __name__ == '__main__':
    # seq = sequence_to_onehot('ATGC')
    # print(len(seq))
    #CNN_model()
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    training_data = pd.read_csv('/home1/pansj/Small_protein_prediction/training_data_whole.csv',index_col=0).values
    data = training_data[:,0]
    label = training_data[:,1]
    label = label.reshape(len(label),-1)
    label[label == -1] = 0
    label = np.array([y[0] for y in label]).reshape(len(label),1)
    data_one_hot = []
    print('start')
    for seq in data:
        seq_one_hot = sequence_to_onehot(seq)
        data_one_hot.append(seq_one_hot)
    #print(np.array(data_one_hot).shape)
    train_data = np.array(data_one_hot)
    skf = StratifiedKFold(n_splits=5,shuffle=True)

    accuracy = []
    for train_index,test_index in skf.split(train_data,label):
        X_train,Y_train = train_data[train_index] , label[train_index]
        X_test, Y_test = train_data[test_index] , label[test_index]
        enc = OneHotEncoder()
        Y_train = enc.fit_transform(Y_train).toarray()
        Y_test = enc.fit_transform(Y_test).toarray()

        cnn_model = CNN_model()
        cnn_model.fit(X_train,Y_train,batch_size=64,epochs=5)
        #prediction = cnn_model.predict(X_test)

        #print(prediction)
        score = cnn_model.evaluate(X_test,Y_test,verbose=0)
        print(score[1])
        #accuracy.append(score)
        accuracy.append(score[1])

    print(accuracy)
    print(np.average(accuracy))






