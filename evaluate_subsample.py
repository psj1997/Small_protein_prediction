import random
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
from CNN_upstream import sequence_to_onehot
from sklearn.preprocessing import OneHotEncoder
from CNN_upstream import UPSTREAM_LENGTH
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from CNN_upstream import CNN


base_root = '/home1/pansj/Small_protein_prediction/'




MAX_SEQ_LENGTH = 354
Gene_length = 153
UPSTREAM_LENGTH = 201




def prepare_data():


    for i in range(5):
        seq_name = []
        for seq_record in SeqIO.parse('/home1/pansj/Small_protein_prediction/CV_testing_{}.fsa'.format(i+1), "fasta"):
                #print(seq_record.id)
                seq_name.append(seq_record.id)

        slice = random.sample(seq_name, len(seq_name)//15)
        sub_sample_list = []
        for seq_record in SeqIO.parse('/home1/pansj/Small_protein_prediction/CV_testing_{}.fsa'.format(i+1), "fasta"):
            if seq_record.id in slice:
                rec1 = SeqRecord(Seq(str(seq_record.seq)), id=str(seq_record.id),description='')
                sub_sample_list.append(rec1)
        print(len(sub_sample_list))

        SeqIO.write(sub_sample_list, base_root + 'subsample_testing/subsample_testing_{}.fna'.format(i+1), 'fasta')

def length(index):
    fw = open("/home1/pansj/Small_protein_prediction/subsample_testing/cv{}/t".format(index), 'w')
    for seq_record in SeqIO.parse('/home1/pansj/Small_protein_prediction/CV_training_1.fsa', "fasta"):
        length = len(str(seq_record.seq).strip('\n'))
        fw.write(str(length))
        fw.write('\n')
    fw.close()

def fix_name(cv_index):
    index = {}
    num = 0
    for line in open("/home1/pansj/Small_protein_prediction/subsample_testing/cv{}/ref.names.list".format(cv_index)):
        temp = str(line).strip('\n').split('\t')
        temp_ = temp[1]+' '+temp[2]
        num += 1
        index[num] = temp_
    tmp1 = open("/home1/pansj/Small_protein_prediction/subsample_testing/cv{}/tmp1".format(cv_index), 'w')
    for line in open("/home1/pansj/Small_protein_prediction/subsample_testing/cv{}/tmp".format(cv_index)):
        temp = str(line).strip('\n').split('\t')
        seq_num = int(temp[1].split('|')[2])
        name = index.get(seq_num+1)
        temp[1] = name
        s=''
        for tempstr in temp:
            s += str(tempstr)+' '
        tmp1.write(str(s))
        tmp1.write('\n')

def swipe():
    for i in range(5):
        os.system('grep ">" '+base_root+'CV_training_'+str(i+1)+'.fsa | awk \'{print "gnl|BL_ORD_ID|"NR-1"\t"$1}\' | sed \'s/>//g\' > '
                  +base_root+'subsample_testing/cv{}/ref.names.list'.format(i+1))
        length(i+1)
        path = base_root+'subsample_testing/cv{}/'.format(i+1)
        #os.system('cd /home1/pansj/Small_protein_prediction/subsample_testing/cv{}'.format(i+1))
        os.system('paste -d\'\t\' '+path+'ref.names.list '+path+'t > '+path+'t2')
        os.system('mv '+path+'t2 '+path +'ref.names.list')
        os.system('awk \'$3 >= 80 && $11 <= 1e-5\' '+path+'swipe_'+str(i+1)+' > '+path+'tmp')
        fix_name(i+1)
        os.system('awk \'$1 != $2 {print $5/$3"\t"$0}\' '+path+'tmp1 | awk \'$1 >= 0.9\' > '+path+'tmp2')

        query = []
        test_new = []
        for line in open(path + "tmp2"):
            temp = str(line).strip('\n').split(' ')
            query.append(temp[0].split('\t')[1])
        query = set(query)
        print(len(query))
        for seq_record in SeqIO.parse(base_root+'subsample_testing/subsample_testing_{}.fna'.format(i + 1), "fasta"):
            # print(str(seq_record.id))
            id_ = str(seq_record.id).strip('\n')
            if id_ not in query:
                rec1 = SeqRecord(Seq(str(seq_record.seq).strip('\n')), id=id_,
                                 description=id_)
                test_new.append(rec1)

        print(len(test_new))
        SeqIO.write(test_new, path + 'CV_testing_swipe.fna', 'fasta')



def train():
    accuracy = []
    recall = []
    precision = []
    f1 = []

    for i in range(5):
        training_x = []
        training_y = []
        for seq_record in SeqIO.parse('/home1/pansj/Small_protein_prediction/CV_training_{}.fsa'.format(i+1), "fasta"):
            training_x.append(str(seq_record.seq).strip('\n'))
            training_y.append(int(str(seq_record.id).split('_')[1][1]))

        testing_x = []
        testing_y = []
        for seq_record in SeqIO.parse('/home1/pansj/Small_protein_prediction/subsample_testing/cv{}/CV_testing_swipe.fna'.format(i+1), "fasta"):
            testing_x.append(str(seq_record.seq).strip('\n'))
            testing_y.append(int(str(seq_record.id).split('_')[1][1]))

        training_y = np.array(training_y).reshape(len(training_y),1)
        testing_y = np.array(testing_y).reshape(len(testing_y),1)

        training_data = []
        testing_data = []
        for seq in training_x:
            seq_one_hot = sequence_to_onehot(seq)
            training_data.append(seq_one_hot)
        training_data = np.array(training_data)

        for seq in testing_x:
            seq_one_hot = sequence_to_onehot(seq)
            testing_data.append(seq_one_hot)
        testing_data = np.array(testing_data)

        print('test length:', np.sum(testing_y == 1), np.sum(testing_y == 0))

        enc = OneHotEncoder()
        Y_train = enc.fit_transform(training_y).toarray()
        Y_test_ = enc.fit_transform(testing_y).toarray()
        print(training_data.shape, Y_train.shape, testing_data.shape, Y_test_.shape)
        X_train_upstream = training_data[:, :(UPSTREAM_LENGTH + 18), :]
        X_train_gene = training_data[:, (UPSTREAM_LENGTH):, :]
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