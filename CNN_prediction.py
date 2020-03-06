from keras.models import Sequential, load_model, model_from_config,Input,Model
from keras.layers.merge import Concatenate
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split,StratifiedKFold,GroupKFold
import json
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard

MAX_SEQ_LENGTH = 354

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
    seq_one_hot = np.array(seq_one_hot).reshape(MAX_SEQ_LENGTH,4)
    return seq_one_hot

from keras.layers import Conv1D,MaxPool2D,MaxPool1D,Flatten,Dense,Dropout,LSTM

def feature_extractor(inp,kernel_size,num_filter):
    out = Conv1D(filters=num_filter,kernel_size=kernel_size,strides=1,activation='relu',padding='same', data_format='channels_last',input_shape=(MAX_SEQ_LENGTH,4))(inp)
    out = MaxPool1D(pool_size=5,strides=1,padding='same')(out)
    out = Flatten()(out)
    return out


def CNN_model_oneconv(kernel_size,num_filter):
    model = Sequential()
    model.add(Conv1D(filters=num_filter,kernel_size=kernel_size,strides=1,activation='relu',padding='valid', data_format='channels_last',input_shape=(MAX_SEQ_LENGTH,4)))
    model.add(MaxPool1D(pool_size=5,strides=1,padding='valid'))
    model.add(Flatten())
    model.add(Dense(64,activation='relu'))
    model.add(Dense(2,activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.summary()
    return model

def CNN_model_multi_conv(kernel_size,num_filter):
    inp = Input(shape=(MAX_SEQ_LENGTH,4))
    conv_block = []
    if len(kernel_size) > 1:
        for size in kernel_size:
            out = feature_extractor(inp,size,num_filter)
            conv_block.append(out)
        conv_out = Concatenate()(conv_block)
    else:
        conv_out = feature_extractor(inp, kernel_size, num_filter)
    Dense_out = Dense(512,activation='relu')(conv_out)
    Drop_out = Dropout(0.5)(Dense_out)
    Dense_out = Dense(64,activation='relu')(Drop_out)
    Drop_out = Dropout(0.5)(Dense_out)
    model_output = Dense(2,activation='softmax')(Drop_out)
    model = Model(inp, model_output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model


def feature_extractor_LSTM(inp,kernel_size,num_filter):
    out = Conv1D(filters=num_filter,kernel_size=kernel_size,strides=1,activation='relu',padding='same', data_format='channels_last',input_shape=(MAX_SEQ_LENGTH,4))(inp)
    out = MaxPool1D(pool_size=5,strides=1,padding='same')(out)
    #out = Flatten()(out)
    return out


def CNN_lstm_model(kernel_size,num_filter):
    inp = Input(shape=(MAX_SEQ_LENGTH,4))
    conv_block = []
    if len(kernel_size) > 1:
        for size in kernel_size:
            out = feature_extractor_LSTM(inp,size,num_filter)
            conv_block.append(out)
        conv_out = Concatenate()(conv_block)
    else:
        conv_out = feature_extractor_LSTM(inp, kernel_size, num_filter)
    lstm_out = LSTM(128, dropout=0.4, recurrent_dropout=0.4, return_sequences=False)(conv_out)
    #lstm_out = LSTM(64, recurrent_dropout=0.4)(lstm_out)
    Drop_out = Dropout(0.5)(lstm_out)
    model_output = Dense(2,activation='softmax')(Drop_out)
    model = Model(inp, model_output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model




def Cross_validation(is_group = False):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    training_data = pd.read_csv('/home1/pansj/Small_protein_prediction/training_upstream_cluster.csv',index_col=0).values
    print(training_data.shape)
    data = training_data[:, 0]
    label = training_data[:, 1]
    cluster = training_data[:, 3]
    print('cluster:', len(set(cluster)))
    label = label.reshape(len(label), -1)
    print(np.sum(label > 0), len(label))
    label[label == -1] = 0
    label = np.array([y[0] for y in label]).reshape(len(label), 1)
    data_one_hot = []
    print('start')
    for seq in data:
        seq_one_hot = sequence_to_onehot(seq)
        data_one_hot.append(seq_one_hot)
    # print(np.array(data_one_hot).shape)
    train_data = np.array(data_one_hot)

    accuracy = []
    recall = []
    precision = []
    f1 = []

    if is_group == False:
        print('CV')

        skf = StratifiedKFold(n_splits=5, shuffle=True)
        count = 0
        for train_index, test_index in skf.split(train_data, label):
            count += 1
            X_train, Y_train = train_data[train_index], label[train_index]
            X_test, Y_test = train_data[test_index], label[test_index]
            enc = OneHotEncoder()
            Y_train = enc.fit_transform(Y_train).toarray()
            Y_test_ = enc.fit_transform(Y_test).toarray()

            tbCallBack = TensorBoard(log_dir='/home1/pansj/Small_protein_prediction/logs_{}'.format(count),  # log 目录
                                     histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
                                     #                  batch_size=32,     # 用多大量的数据计算直方图
                                     write_graph=True,  # 是否存储网络结构图
                                     write_grads=True,  # 是否可视化梯度直方图
                                     write_images=True,  # 是否可视化参数
                                     embeddings_freq=0,
                                     embeddings_layer_names=None,
                                     embeddings_metadata=None)

            cnn_model = CNN_lstm_model([3, 4, 5], 32)

            history = cnn_model.fit(X_train, Y_train, batch_size=128, epochs=10,workers=8,validation_data=(X_test,Y_test_),callbacks=[tbCallBack])
            print(history.history)


            pre_y = []
            pre = cnn_model.predict(X_test)
            for i in range(len(pre)):
                if pre[i][0] > pre[i][1]:
                    pre_y.append(0)
                else:
                    pre_y.append(1)
            pre_y = np.array(pre_y).reshape(len(pre_y), 1)
            accuracy.append(accuracy_score(Y_test, pre_y))
            recall.append(recall_score(Y_test, pre_y))
            precision.append(precision_score(Y_test, pre_y))
            f1.append(f1_score(Y_test, pre_y))
    else:
        print('group_cv')
        skf = GroupKFold(n_splits=5)
        count = 0
        for train_index, test_index in skf.split(train_data, label, cluster):
            count += 1
            print(len(train_index), len(test_index))
            X_train, Y_train = train_data[train_index], label[train_index]
            X_test, Y_test = train_data[test_index], label[test_index]
            enc = OneHotEncoder()
            Y_train = enc.fit_transform(Y_train).toarray()
            Y_test_ = enc.fit_transform(Y_test).toarray()

            cnn_model = CNN_model_multi_conv([3,4,5], 32)

            tbCallBack = TensorBoard(log_dir='/home1/pansj/Small_protein_prediction/CNN_group/cnn_results_upstream_logs_{}'.format(count),  # log 目录
                                     histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
                                     #                  batch_size=32,     # 用多大量的数据计算直方图
                                     write_graph=True,  # 是否存储网络结构图
                                     write_grads=True,  # 是否可视化梯度直方图
                                     write_images=True,  # 是否可视化参数
                                     embeddings_freq=0,
                                     embeddings_layer_names=None,
                                     embeddings_metadata=None)

            history = cnn_model.fit(X_train, Y_train, batch_size=128, epochs=20,workers=8,validation_data=(X_test, Y_test_),callbacks=[tbCallBack])
            print(history.history)


            pre_y = []
            pre = cnn_model.predict(X_test)
            for i in range(len(pre)):
                if pre[i][0] > pre[i][1]:
                    pre_y.append(0)
                else:
                    pre_y.append(1)
            pre_y = np.array(pre_y).reshape(len(pre_y), 1)
            accuracy.append(accuracy_score(Y_test, pre_y))
            recall.append(recall_score(Y_test, pre_y))
            precision.append(precision_score(Y_test, pre_y))
            f1.append(f1_score(Y_test, pre_y))

    print('CNN CV_group_new: kernel size:[3,4,5] ; num_kernel = 32 epoches = 20;')
    print(accuracy)
    print('accuracy: ', np.average(accuracy))
    print(recall)
    print('recall:', np.average(recall))
    print(precision)
    print('precision:', np.average(precision))
    print(f1)
    print('f1:', np.average(f1))


if __name__ == '__main__':
    #CNN_lstm_model([3,4,5],32)
    Cross_validation(is_group=True)






