from keras.models import Sequential, load_model, model_from_config,Input,Model
from keras.layers.merge import Concatenate
from keras.layers import Conv1D,MaxPool2D,MaxPool1D,Flatten,Dense,Dropout,LSTM,AveragePooling1D,ReLU
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,StratifiedKFold,GroupKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score
from keras.utils.vis_utils import plot_model
from keras.layers.normalization import BatchNormalization

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO


MAX_SEQ_LENGTH = 354
Gene_length = 153
UPSTREAM_LENGTH = 201

def extract_feature_gene(inp,num_filter,kernel_size,conv_stride,pool_size,pool_stride):
    out = Conv1D(filters=num_filter,kernel_size=kernel_size,strides=conv_stride,activation='relu',padding='valid',
                 data_format='channels_last')(inp)
    #out = BatchNormalization()(out)
    #out = ReLU()(out)
    out = AveragePooling1D(pool_size=pool_size,strides=pool_stride,padding='valid')(out)
    #out = BatchNormalization()(out)
    out = Flatten()(out)
    return out

def extract_feature_upstream(inp,num_filter,kernel_size,conv_stride,pool_size,pool_stride):
    out = Conv1D(filters=num_filter,kernel_size=kernel_size,strides=conv_stride,activation='relu',padding='valid',
                 data_format='channels_last')(inp)
    #out = BatchNormalization()(out)
    #out = ReLU()(out)
    out = MaxPool1D(pool_size=pool_size,strides=pool_stride,padding='valid')(out)
    #out = BatchNormalization()(out)
    out = Flatten()(out)
    return out

def CNN(size_gene,num_gene,gene_stride,gene_pool,gene_pool_stride,
        size_upstream,num_upstream,upstream_stride,upstream_pool,upstream_pool_stride):
    inp_gene = Input(shape=(Gene_length,4))
    inp_upstream = Input(shape=(UPSTREAM_LENGTH+18, 4))
    conv_block = []
    if len(size_gene) > 1:
        for size in size_gene:
            out = extract_feature_gene(inp_gene,num_gene,size,gene_stride,gene_pool,gene_pool_stride)
            conv_block.append(out)
        conv_out = Concatenate()(conv_block)
    else:
        conv_out = extract_feature_gene(inp_gene,num_gene,size_gene,gene_stride,gene_pool,gene_pool_stride)

    upstream_conv_block = []
    if len(size_upstream) > 1:
        for size in size_upstream:
            out = extract_feature_upstream(inp_upstream,num_upstream,size,upstream_stride,upstream_pool,upstream_pool_stride)
            upstream_conv_block.append(out)
        conv_out_upstream = Concatenate()(upstream_conv_block)
    else:
        conv_out_upstream = extract_feature_upstream(inp_upstream,num_upstream,size_upstream,
                                            upstream_stride,upstream_pool,upstream_pool_stride)
    output_list = [conv_out,conv_out_upstream]
    output_all = Concatenate()(output_list)

    Dense_out = Dense(512,activation='relu')(output_all)
    #Dense_out = BatchNormalization()(Dense_out)
    #Dense_out = ReLU()(Dense_out)
    Drop_out = Dropout(0.5)(Dense_out)
    Dense_out = Dense(64,activation='relu')(Drop_out)
    #Dense_out = BatchNormalization()(Dense_out)
    #Dense_out = ReLU()(Dense_out)
    Drop_out = Dropout(0.5)(Dense_out)
    model_output = Dense(2, activation='softmax')(Drop_out)
    model = Model([inp_gene,inp_upstream], model_output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #plot_model(model,to_file='CNN_model.jpg',show_shapes=True,show_layer_names=True)
    model.summary()
    return model

def sequence_to_onehot(seq):
    """
    return the one_hot of the sequence 4*m A(1,0,0,0) C(0,1,0,0) G(0,0,1,0) T(0,0,0,1)
    :param seq:
    :param max_length:
    :return:
    """
    seq_one_hot = []
    for i in range(MAX_SEQ_LENGTH):
        if i < len(seq):
            if seq[i] == 'A':
                seq_one_hot.append([1,0,0,0])
            elif seq[i] == 'C':
                seq_one_hot.append([0,1,0,0])
            elif seq[i] == 'G':
                seq_one_hot.append([0,0,1,0])
            elif seq[i] == 'T':
                seq_one_hot.append([0,0,0,1])
            else:
                seq_one_hot.append([0,0,0,0])
        else:
            seq_one_hot.append([0,0,0,0])
    seq_one_hot = np.array(seq_one_hot).reshape(MAX_SEQ_LENGTH,4)
    return seq_one_hot

def train():
    training_x = []
    training_y = []
    for seq_record in SeqIO.parse('/home1/pansj/Small_protein_prediction/CV_training_1.fsa', "fasta"):
        training_x.append(str(seq_record.seq).strip('\n'))
        training_y.append(int(str(seq_record.id).split('_')[1][1]))

    testing_x = []
    testing_y = []
    for seq_record in SeqIO.parse('/home1/pansj/Small_protein_prediction/CV_testing1_1.fsa', "fasta"):
        testing_x.append(str(seq_record.seq).strip('\n'))
        testing_y.append(int(str(seq_record.id).split('_')[1][1]))

    #training_x = np.array(training_x)
    training_y = np.array(training_y).reshape(len(training_y),1)
    #testing_x = np.array(testing_x)
    testing_y = np.array(testing_y).reshape(len(testing_y),1)

    training_data = []
    testing_data = []
    for seq in training_x:
        seq_one_hot = sequence_to_onehot(seq)
        training_data.append(seq_one_hot)
    train_data = np.array(training_data)

    for seq in testing_x:
        seq_one_hot = sequence_to_onehot(seq)
        testing_data.append(seq_one_hot)
    testing_data = np.array(testing_data)

    accuracy = []
    recall = []
    precision = []
    f1 = []

    print('test length:', np.sum(testing_y == 1), np.sum(testing_y == 0))

    enc = OneHotEncoder()
    Y_train = enc.fit_transform(training_y).toarray()
    Y_test_ = enc.fit_transform(testing_y).toarray()
    print(train_data.shape, Y_train.shape, testing_data.shape, Y_test_.shape)
    X_train_upstream = train_data[:, :(UPSTREAM_LENGTH + 18), :]
    X_train_gene = train_data[:, (UPSTREAM_LENGTH):, :]
    X_test_upstream = testing_data[:, :(UPSTREAM_LENGTH + 18), :]
    X_test_gene = testing_data[:, (UPSTREAM_LENGTH):, :]

    print(X_train_upstream.shape, X_train_gene.shape)

    cnn_model = CNN(size_gene=[3, 4, 5], num_gene=32, gene_stride=3, gene_pool=6, gene_pool_stride=3
                    , size_upstream=[16], num_upstream=128, upstream_stride=3, upstream_pool=6, upstream_pool_stride=3)

    cnn_model.fit([X_train_gene, X_train_upstream], Y_train, batch_size=128, epochs=10, workers=8,
                  use_multiprocessing=True, class_weight={0: 1, 1: 10})

    pre = cnn_model.predict([X_test_gene, X_test_upstream])
    pre_y = [0 if pre[i][0] > pre[i][1] else 1 for i in range(len(pre))]
    pre_y = np.array(pre_y).reshape(len(pre_y), 1)
    accuracy.append(accuracy_score(testing_y, pre_y))
    recall.append(recall_score(testing_y, pre_y))
    precision.append(precision_score(testing_y, pre_y))
    f1.append(f1_score(testing_y, pre_y))

    print(accuracy)
    print('accuracy: ', np.average(accuracy))
    print(recall)
    print('recall:', np.average(recall))
    print(precision)
    print('precision:', np.average(precision))
    print(f1)
    print('f1:', np.average(f1))


train()