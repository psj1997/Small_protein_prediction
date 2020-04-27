"""
All function process the original sequence data
"""
import pandas as pd
import os
from Bio import SeqIO
from numba import jit
from itertools import chain
import numpy as np
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from multiprocessing import Pool
from functools import partial

root_path = '/home1/pansj/Small_protein_prediction/'
path_fna = '/home1/pansj/small_protein/'
gbk_path = '/home1/pansj/small_protein/gbk/'
def SRS_list():
    """
    return all 1773 metagenomic sample
    """
    data = pd.read_excel(root_path+'1-s2.0-S0092867419307810-mmc3.xlsx')
    SRS = data['Sample ID'].values


    file_fna = os.listdir(path_fna)
    fna = [file[0:-4] for file in file_fna if file[-4:] == '.fna' and file[0:-4] in SRS]
    return fna


def gene_coords_list(sample):
    """
    return all coords of genes : sample_id  contig_id gene_id start_position end_position
    """
    gene_coords = {}
    gene_coords[sample] = {}
    gene_number = 0
    for line in open(gbk_path+sample):
        if line[0:10] == 'DEFINITION':
            contig = line.split(';')[2].split('=')[1][1:-1]
            gene_coords[sample][contig] = {}
            gene_number = 0

        if line.strip().split(' ')[0] == 'CDS':
              gene_number += 1
              gene_coords[sample][contig][gene_number] = line.strip().split(' ')[-1]

@jit
def read_sequence(sample_id,contig_id,start,end,is_complememt,upstream=0):
    """
    read the sequence from the fna file
    """
    start = int(start)
    end = int(end)
    replace_dict = {'A':'T','T':'A','G':'C','C':'G'}
    if is_complememt is False:
        if start > upstream:
            for seq_record in SeqIO.parse(path_fna + '{}.fna'.format(sample_id),"fasta"):
                    if seq_record.id == str(contig_id):
                        temp = str(seq_record.seq).strip('\n')
                        return temp[start-(1+upstream):end]
        return None

    else:
        for seq_record in SeqIO.parse(path_fna + '{}.fna'.format(sample_id),"fasta"):
            if seq_record.id == str(contig_id):
                temp = str(seq_record.seq).strip('\n')
                if len(temp) > end + upstream:
                    temp = temp[start-1:end+upstream][::-1]
                    temp = list(temp)
                    re_temp = [replace_dict[nu] for nu in temp]
                    temp = ''.join(re_temp)
                    return temp
                else:
                    return None
        return None

@jit
def prepare_samll_orf(sample,upstream=0):
    """
    find all small_orf 15bp~153bp from contig
    """
    whole_list = []
    coords_gbk = gene_coords_list(sample)
    for key_contig in coords_gbk[sample]:
        for gene_id in coords_gbk[sample][key_contig]:
            gene_description = coords_gbk[sample][key_contig][gene_id]
            if '>' not in gene_description and '<' not in gene_description \
                    and 'complement' not in gene_description:
                temp_gene = gene_description.split('.')
                start = int(temp_gene[0])
                end = int(temp_gene[-1])
                if start > end:
                    start, end = end, start
                gene_length = np.abs(int(start) - int(end)) + 1
                if gene_length <= 153:
                    sequence = read_sequence(sample, key_contig, start, end, False,upstream=upstream)
                    if sequence != None:
                        rec1 = SeqRecord(Seq(str(sequence)), id=sample + '_' + str(key_contig) + '_' + str(gene_id)
                         + '_' + gene_description ,description='')
                        whole_list.append(rec1)
            elif '>' not in gene_description and '<' not in gene_description \
                    and 'complement' in gene_description:
                temp_gene = gene_description[11:-1]
                temp_gene = temp_gene.split('.')
                start = int(temp_gene[0])
                end = int(temp_gene[-1])
                if start > end:
                    start, end = end, start
                gene_length = np.abs(int(start) - int(end)) + 1

                if gene_length <= 153:
                    sequence = read_sequence(sample, key_contig, start, end, True,upstream=upstream)
                    if sequence != None:
                        rec1 = SeqRecord(Seq(str(sequence)), id=sample + '_' + str(key_contig) + '_' + str(gene_id)
                         + '_' + gene_description , description='')
                        whole_list.append(rec1)
    print(len(whole_list))
    SeqIO.write(whole_list, root_path+'upstream/whole_seq_{}_upstream.fna'.format(sample), 'fasta')

def multiple_smorf():
    """
    mupti process to find small_orf
    """
    sample_list = SRS_list()
    pool = Pool(20)
    partial_work = partial(prepare_samll_orf, upstream=0)
    pool.map(partial_work, sample_list)
    pool.close()
    pool.join()


def label_data():
    """
    label every data using labels from the Paper
    """
    positive_list = {}
    sample_list = SRS_list()
    for line in open(root_path+'Data File 1.faa'):
        if line[0:3] == 'DNA':
            break
        if line[0] == '>':
            temp = line.split('_')
            if temp[-3] == 'prediction':
                contig_id = temp[-2]
                sample_id = temp[-7]
            if temp[-4] == 'prediction':
                contig_id = temp[-3] + '_' + temp[-2]
                sample_id = temp[-8]
            gene_id = temp[-1]
            if sample_id not in positive_list:
                positive_list[sample_id] = {}
            if contig_id not in positive_list[sample_id]:
                positive_list[sample_id][contig_id] = []
            positive_list[sample_id][contig_id].append(int(gene_id))

    num_positive = 0
    num_negative = 0
    whole_list = []
    for sample in sample_list:
        for seq_record in SeqIO.parse(
                root_path + 'upstream/whole_seq_{}_upstream.fna'.format(sample), "fasta"):
                temp = seq_record.description.split('_')
                srs = temp[0]
                if len(temp) == 4:
                    contig = temp[1]
                else:
                    contig = temp[1] + '_' + temp[2]
                gene = temp[-2]
                if srs in positive_list:
                    if contig in positive_list[srs]:
                        if int(gene) in positive_list[srs][contig]:
                            num_positive += 1
                            rec1 = SeqRecord(Seq(str(seq_record.seq).strip('\n')), id='+',
                                             description=seq_record.description)
                            whole_list.append(rec1)
                if srs not in positive_list or contig not in positive_list[srs] or int(gene) not in positive_list[srs][
                    contig]:
                    num_negative += 1
                    rec1 = SeqRecord(Seq(str(seq_record.seq).strip('\n')), id='-', description=seq_record.description)
                    whole_list.append(rec1)
    print(num_negative)
    print(num_positive)
    print(len(whole_list))
    SeqIO.write(whole_list, root_path + 'whole_upstream.fna','fasta')

def rm_same():
    """
    remove all sample sequence from the data
    """
    positive_list = []
    negative_list = []

    for seq_record in SeqIO.parse(root_path + 'whole_upstream.fna', 'fasta'):
        id = seq_record.id
        sequence = str(seq_record.seq).strip('\n')

        if id == '-':
            negative_list.append(sequence)
        if id == '+':
            positive_list.append(sequence)

    data_whole = []

    positive_seq = set(positive_list)
    print('positive:', len(positive_seq))

    negative_seq = set(negative_list)
    print('negative:', len(negative_seq))
    num = 0

    for seq in positive_seq:
        if seq not in negative_seq:
            num += 1
            rec1 = SeqRecord(Seq(str(seq).strip('\n')), id='sequence_' + str(num), description='+')
            data_whole.append(rec1)

    print(num)

    for seq in negative_seq:
        if seq not in positive_seq:
            num += 1
            rec1 = SeqRecord(Seq(str(seq).strip('\n')), id='sequence_' + str(num), description='-')
            data_whole.append(rec1)

    print(num)
    SeqIO.write(data_whole, root_path + 'whole_upstream_rmsame.fna', 'fasta')


def prepare_csv():
    sequence = []
    label = []
    num_seq = []
    for seq_record in SeqIO.parse(root_path + 'whole_upstream_rmsame.fna','fasta'):
        description = str(seq_record.description)
        seq = str(seq_record.seq).strip('\n')
        label_seq = description.split(' ')[-1]
        id_ = str(seq_record.id)
        if label_seq == '+':
            sequence.append(seq)
            label.append(1)
            num_seq.append(id_)
        else:
            sequence.append(seq)
            label.append(0)
            num_seq.append(id_)
    sequence = np.array(sequence).reshape(len(sequence),1)
    label = np.array(label).reshape(len(label),1)
    num_seq = np.array(num_seq).reshape(len(num_seq),1)
    data = np.concatenate((sequence,label),axis=1)
    data = np.concatenate((data,num_seq),axis=1)
    data = pd.DataFrame(data,columns=['sequence','label','number'])
    print(len(sequence))
    data.to_csv('whole_upstream_rmsame.csv')

def linclust():
    """
    cluster using lincust
    """
    os.system('./mmseqs createdb whole_upstream_rmsame.fna upstream')
    os.system('mkdir tmp')
    os.system('./mmseqs linclust upstream upstream_clu tmp --min-seq-id 0.5 -c 0.5')
    os.system('./mmseqs createtsv upstream upstream upstream_clu upstream_cluster.tsv')


def prepare_cluster():
    """
    get the csv file with cluster
    """
    cluster = pd.read_csv('upstream_cluster.tsv',sep='\t',header=None,names=['cluster','number'])
    data = pd.read_csv('whole_upstream_rmsame.csv',sep=',',header=0,index_col=0)
    cluster_data = pd.merge(data,cluster)
    cluster_data.to_csv('whole_upstream_cluster.csv')



if __name__ == '__main__':
    """
    Get all smorf
    label smorf
    remvoe same sequence
    Get the cluster and Using the cluster.csv to test Cross Validation later
    """
    multiple_smorf()
    label_data()
    rm_same()
    prepare_csv()
    linclust()
    prepare_cluster()

