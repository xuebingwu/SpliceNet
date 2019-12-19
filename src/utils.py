from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D as Conv2D
from keras.layers import MaxPooling2D
from keras import backend as K
import tensorflow as tf
import numpy
import random
import pickle

import os, sys, copy, fnmatch

from os import path

import time
import datetime

# map letters to numbers
letter2num = {'A': 0, 'C': 1, 'G': 2, 'T': 3}


# return a time string
def time_string():
    return datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')


#
def sparsity_regularizer(x):
    return 0.05 * K.abs(tf.math.zero_fraction(x) - 0.60)  # optimal: 60% of elements are 0
    # return 0.01 * K.sum(K.square(x)) # l2, but may be not. doesn't work
    # return 0.01 * K.sum(K.square(x))*(1-tf.math.zero_fraction(x)) # combined


# get file length
def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


# encode a single sequence
def encode_seq(seq):
    d = numpy.zeros((4, len(seq)), numpy.int8)
    for i in range(len(seq)):
        d[letter2num[seq[i]], i] = 1
    return d


def encode_seqs(seqs):
    eseqs = numpy.zeros((len(seqs), 4, len(seqs[0])), numpy.int8)
    # random sequence 
    for i in range(len(seqs)):
        eseqs[i, :, :] = encode_seq(seqs[i])
    return eseqs


# or may be should pad with all 0.25?
# the problem with 0 is that it avoids penalty if the pwm has negative values    
def pad_encoded_seqs(eseqs, N):
    # add N 0 column on each side
    new_seqs = numpy.zeros((len(eseqs), 4, eseqs.shape[2] + N + N)) + 0.25
    new_seqs[:, :, N:N + eseqs.shape[2]] = eseqs
    return new_seqs


def reverse_encode_seq(m):
    # convert a matrix to sequence
    ab = 'ACGT'
    seq = ''
    for i in range(m.shape[1]):
        seq = seq + ab[numpy.where(m[:, i] == 1)[0][0]]
    return seq


def reverse_encode_seqs(m):
    seqs = []
    for i in range(m.shape[0]):
        seqs.append(reverse_encode_seq(m[i, :, :]))
    return seqs


def write_fasta(seqs, filename):
    f = open(filename, 'w')
    for i in range(len(seqs)):
        f.write('>' + str(i) + '\n')
        f.write(seqs[i] + '\n')
    f.close()


# concat new seqs into sequences of another file
# use this to combine sequences in multiple regions
def concat_fasta(seqs, filename):
    fin = open(filename, 'r')
    fout = open(filename + 'tmp', 'w')
    for i in range(len(seqs)):
        fout.write(fin.readline())
        fout.write(fin.readline().strip() + seqs[i] + '\n')
    fin.close()
    fout.close()
    os.system('mv ' + filename + 'tmp ' + filename)


# encoding N random sequences of length L
def random_encoded_seqs(N, L):
    x = numpy.zeros((N, 4, L), numpy.int8)
    # random sequence 
    a = numpy.random.random_integers(0, 3, size=(N, L))
    for i in range(N):
        for j in range(L):
            x[i, a[i, j], j] = 1
    return x


# shuffle expression data for each RBP, to generate new experiments where RBP co-expression are disrupted
# based on TSNE plot, doesn't look like this completely removes clustering
def shuffle_RBP_expr(
        RBP_expr
):
    # for each row, randomly shuffle its elements/columns
    for i in range(RBP_expr.shape[0]):
        numpy.random.shuffle(RBP_expr[i,])


# positional effect: how does motif in each region contribute to splicing outcome
def generate_positional_effect(
        n_motif,
        n_region,
        effect_scale,
        fraction_functional  # set small effect to 0
):
    positional_effect = numpy.random.uniform(low=-1, high=1, size=(n_motif * n_region, 1))

    # make sure the sum is close to 0, so that overall splicing will not bias towards 0 or 1
    while abs(numpy.sum(positional_effect)) > 0.001:
        positional_effect = numpy.random.uniform(low=-1, high=1, size=(n_motif * n_region, 1))

    # set small effect to 0: 
    positional_effect[abs(positional_effect) < (1 - fraction_functional)] = 0

    return positional_effect * effect_scale / ((n_region * n_motif) ** 0.5)


def generate_input_sequences(
        n_region,
        n_exon_train,
        l_seq
):
    seqs = []
    for i in range(n_region):
        seqs.append(random_encoded_seqs(n_exon_train, l_seq).reshape(n_exon_train, 4, l_seq, 1))

    return seqs


# generate training data from given sequences
def generate_training_data(
        seqs,  # cannot be empty
        input_seqs,  # if empty, start from seqs
        RBP_expr,
        model0,
        gamma_shape,
        gamma_scale,
        n_experiment,
        group_by,
        remove_non_regulated,
        index
):
    if seqs == []:
        print(time_string(), "ERROR: seqs empty. cannot generate training data")
        raise

    n_region = 0
    while model0.layers[n_region].get_weights() == []:
        n_region = n_region + 1
    n_motif = model0.layers[n_region].get_weights()[0].shape[3]
    n_exon = int(seqs[0].shape[0])

    # generate sequence matrix
    if len(input_seqs) == 0:
        input_seqs = [[]] * n_region
        for i in range(n_region):
            input_seqs[i] = numpy.tile(seqs[i], (n_experiment, 1, 1, 1))

    if len(RBP_expr) == 0:
        # if no RBP_expr provided, generate random ones
        expression = numpy.random.gamma(gamma_shape, gamma_scale, size=(n_motif, n_experiment))
    else:
        # take from RBP_expr
        expression = RBP_expr[:n_motif, index: index + n_experiment]
        index = index + n_experiment

    input_RBP_expr = numpy.zeros((n_exon * n_experiment, n_motif))
    for i in range(n_experiment):
        input_RBP_expr[(i * n_exon):(i + 1) * n_exon, :] = numpy.tile(expression[:, i], [n_exon, 1])
    x = input_seqs.copy()
    x.append(input_RBP_expr)

    # group by
    if group_by == 'random':
        print(time_string(), "group data randomly")
        idx = numpy.arange(n_exon * n_experiment)
        numpy.random.shuffle(idx)
        for i in range(n_region + 1):
            x[i] = x[i][idx, :]

    y = model0.predict(x)

    # TODO: remove non-informative data
    # sometimes due to rare motif occurance some sequence will have no match to any motif, their PSI will be 0.5

    if remove_non_regulated:
        print(time_string(), "remove non-regulated exons")
        sel1 = numpy.where(abs(y - 0.5) < 0.001)

        x2 = [[]] * (n_region + 1)
        for j in range(n_region + 1):
            x2[i] = numpy.delete(x[j], sel1, axis=0)
        y2 = numpy.delete(y, sel1, axis=0)
        return x2, y2, index, input_seqs

    return x, y, index, input_seqs


def get_model_parameters(model):
    n_region = 0
    while model.layers[n_region].get_weights() == []:
        n_region = n_region + 1
    n_motif = model.layers[n_region].get_weights()[0].shape[3]
    l_motif = model.layers[n_region].get_weights()[0].shape[1]
    l_seq = model.layers[0].input_shape[2]
    return n_region, n_motif, l_motif, l_seq


def get_model_parameters_from_weights(weights):
    n_motif = weights[0].shape[3]
    l_motif = weights[0].shape[1]
    n_region = int(weights[2].shape[0] / n_motif)
    return n_region, n_motif, l_motif


#### inforomation content of a pwm
# assume column sum = 1
def info(pwm):
    ic = 2.0 * pwm.shape[1]
    for j in range(pwm.shape[1]):  # each column
        for i in range(pwm.shape[0]):
            if pwm[i, j] > 0:
                ic = ic + pwm[i, j] * numpy.log2(pwm[i, j])
    return ic / pwm.shape[1]


# calculate the information content of motifs given the weight matrix of the conv layer
def information_content(
        weights,  # conv layer only, both kernal and bias
        motifs=[],  # calculate motif rank
        kmers=[]
):
    (a, l_motif, b, n_motif) = weights[0].shape

    # print(time_string(), "generate all possible kmers of length ",l_motif)
    if len(kmers) == 0:
        kmers = Generate_all_kmers(l_motif)

    ekmers = encode_seqs(kmers)
    kmer_shape = (4, l_motif, 1)
    ekmers = ekmers.reshape(ekmers.shape[0], 4, l_motif, 1)

    # print(time_string(),  "scoring short sequences using the first layer")
    model2 = Sequential()

    # layer 1
    model2.add(Conv2D(n_motif, kernel_size=(4, l_motif), input_shape=kmer_shape, weights=weights))
    # get output on input sequences
    activations = model2.predict(ekmers)
    n_kmer = len(kmers)
    kmers = numpy.array(kmers)
    rnk = numpy.zeros(n_motif, dtype=int)
    topkmer = numpy.zeros(n_motif, dtype=int)
    info = numpy.zeros(n_motif)
    # calculate max info
    p = numpy.ones(n_kmer) / n_kmer
    info_max = -sum(p * numpy.log2(p))

    # print(time_string(), "calcualte info")
    for i in range(n_motif):
        score = numpy.exp(activations[:, 0, 0, i])
        p = score / sum(score)
        info[i] = -sum(p * numpy.log2(p))
    info = 1 - info / info_max

    # print(time_string(), "calculate rank")
    for i in range(n_motif):
        am = numpy.argsort(activations[:, 0, 0, i])  # small first
        topkmer[i] = am[-1]
        kmers2 = list(kmers[am])
        if motifs != []:
            rnk[i] = n_kmer - kmers2.index(motifs[i])
    return info, rnk, topkmer, kmers


def layer1_motif(weights, N, alpha, activation_func, output_label):
    # weights=model.layers[0].get_weights()
    # used weights to score N random sequences of length L
    # use those active the neuron 70% of max activation to construct pwm
    # make motif logo using the pwm

    # example
    # layer1_motif(model.layers[0].get_weights(),1000000,0.7,'relu')

    # load weights
    # weights=pickle.load(open( "keras1-model-weights.pickle", "rb" ))
    # import pickle

    # length of the kernal/motif
    l_motif = weights[0].shape[1]

    # number of kernals
    n_motif = weights[0].shape[3]

    print("generate large number of random sequences...")
    rndseqs = random_encoded_seqs(N, l_motif)
    rndseqs_shape = (4, l_motif, 1)
    rndseqs = rndseqs.reshape(rndseqs.shape[0], 4, l_motif, 1)

    # scoring short sequences using the first layer
    model2 = Sequential()

    # layer 1
    model2.add(Conv2D(n_motif, kernel_size=(4, l_motif), activation=activation_func, input_shape=rndseqs_shape,
                      weights=weights))
    # get output on input sequences
    activations = model2.predict(rndseqs)
    # find the max across sequences
    ma = numpy.amax(activations, axis=0)
    # normalize
    nw = activations / ma
    # for each filter, generate pwm for activation > 0.7
    for i in range(n_motif):  # for each filter
        pwm = numpy.sum(rndseqs[nw[:, 0, 0, i] > alpha, :, :], axis=0)
        pwm = pwm[:, :, 0]
        num = numpy.sum(pwm[:, 0])
        if num < 20:
            continue
        pwm = pwm / float(num)
        ic = info(pwm)
        filename = '-'.join([output_label, str(ic), str(i), str(num)])
        numpy.savetxt(filename, pwm, fmt='%f', delimiter='\t')
        # generate logo
        os.system('kpLogo ' + filename + ' -pwm -o ' + filename)
        # os.system('rm '+output_label+'*.pdf')
    os.system('rm ' + output_label + '*.eps')
    os.system('rm ' + output_label + '*freq*')
    os.system('tar zcvf ' + output_label + '.tar.gz ' + output_label + '*')
    os.system('rm ' + output_label + "-*")


def Extend_kmers(kmers):
    kmers2 = []
    for kmer in kmers:
        kmers2.append(kmer + 'A')
        kmers2.append(kmer + 'C')
        kmers2.append(kmer + 'G')
        kmers2.append(kmer + 'T')
    return kmers2


def Generate_all_kmers(k):
    kmers = ['A', 'C', 'G', 'T']
    for i in range(k - 1):
        kmers = Extend_kmers(kmers)
    return kmers


def layer1_motif_rank(model, motifs):
    '''
    # example usage
    from splicenet9 import *
    model = load_model('motif4exon1000expr100l2.best_model.h5')
    with open('motif4exon1000expr100l2-motif.pickle','rb') as f:
        motifs, pe = pickle.load(f)
    rnk = layer1_motif_rank(model,motifs)
    '''

    # get model parameters
    n_region = 0
    while model.layers[n_region].get_weights() == []:
        n_region = n_region + 1
    weights = model.layers[n_region].get_weights()
    n_motif = weights[0].shape[3]
    l_motif = weights[0].shape[1]

    # print("generate all possible kmers of length ",l_motif)

    kmers = Generate_all_kmers(l_motif)

    ekmers = encode_seqs(kmers)

    pad_ekmers = pad_encoded_seqs(ekmers, int(l_motif / 2))

    new_l_motif = l_motif + int(l_motif / 2) * 2
    kmer_shape = (4, new_l_motif, 1)
    pad_ekmers = pad_ekmers.reshape(pad_ekmers.shape[0], 4, new_l_motif, 1)

    # scoring short sequences using the first layer
    model2 = Sequential()

    # layer 1
    model2.add(Conv2D(n_motif, kernel_size=(4, l_motif), input_shape=kmer_shape, weights=weights))
    # get output on input sequences
    activations = model2.predict(pad_ekmers)
    # find the max for each filter
    ma = numpy.amax(activations, axis=2)

    # for each filter, rank all kmers
    n_kmer = len(kmers)
    kmers = numpy.array(kmers)
    rnk = numpy.zeros(n_motif, dtype=int)
    for i in range(n_motif):
        am = numpy.argsort(ma[:, 0, i])  # small first
        kmers2 = list(kmers[am])
        rnk[i] = n_kmer - kmers2.index(motifs[i])
        # find rank of the known motif
        # print top 3 
        # top3score = ma[am[n_kmer-3:n_kmer],0,i]
        # top3motif = kmers[am[n_kmer-3:n_kmer]]
        # print(top3motif)
        # print(top3score)
    return rnk


def motif_score(pwm, encoded_seq):
    # identical length
    return float(numpy.sum(pwm * encoded_seq)) / encoded_seq.shape[1]


def total_motif_score(pwm, encoded_seq, min_score):
    # total motif score for motif match with a minimum score
    # potential problem: for homopolymer motif, will count overlapping ones
    total_score = 0.0
    for i in range(encoded_seq.shape[1] - pwm.shape[1] + 1):
        score = motif_score(pwm, encoded_seq[:, i:(i + pwm.shape[1])])
        if score >= min_score:
            total_score = total_score + score
    return total_score


# similarity between two weight matrix of conv2d layers
def pwm_similarity(w1, w2):
    '''
    # example usage:
    from splicenet10 import *
    model1=load_model('motif20-model-1.h5')
    model2=load_model('motif20-model-2.h5')
    w1=model1.layers[2].get_weights()
    w2=model2.layers[2].get_weights()
    s=pwm_similarity(w1[0][:,:,:,0],w2[0][:,:,:,0])
    '''
    return numpy.corrcoef(w1.flatten(), w2.flatten())[0, 1]


# merge a list of models    
# method: max_motif_info, max_motif_info_concensus, average,
'''
from splicenet10 import *

with open('motif20-motif.pickle', 'rb') as f:
    motifs,positional_effect = pickle.load(f) 

infos = numpy.zeros((10,20))
rnks = numpy.zeros((10,20))
models=[]
for i in range(10):
    #models.append(load_model('motif20-model-'+str(i)+'.h5'))
    info, rnk = information_content(models[i],motifs)
    infos[i,:] = info
    rnks[i,:] = rnk
        
numpy.savetxt('info.txt',infos,fmt='%.4f') 
numpy.savetxt('rnks.txt',rnks,fmt='%d')     
model = merge_models(models,'correlation')
info, rnk = information_content(model,motifs)
model.save('motif20-merge26max.best_model.h5')


from splicenet10 import *

with open('motif10region4-motif.pickle', 'rb') as f:
    motifs,positional_effect = pickle.load(f) 

model,motifs,infos,rnks,merged_info,merged_rnk = merge_models_with_job_name('motif10region4',10,10,'correlation',5)

'''


# model,weights,motifs,infos,rnks,topkmers,merged_info, merged_rnk, first_rnk_count = merge_models_with_job_name('motif100region4*-model-*.h5','motif100region4-motif.pickle')

def merge_models_with_job_name(
        model_filename_pattern,  # motif100region4*-model-*.h5
        motif_filename,  # *-motif.pickle
        method='vote',
        power=1,
        min_info=0,
        compile_model=False):
    print(time_string(), "load motifs")
    motifs = []
    with open(motif_filename, 'rb') as f:
        motifs, positional_effect = pickle.load(f)
    if motifs == []:
        raise SystemExit("ERROR: no *-motif.pickle file found!")

    n_motif = len(motifs)
    print(time_string(), str(n_motif) + " motifs")

    # determine how many models to merge
    n_model = 0
    for filename in os.listdir('.'):
        if fnmatch.fnmatch(filename, model_filename_pattern):
            n_model = n_model + 1

    if n_model < 2:
        raise SystemExit("ERROR: too few models to merge: n_model = " + str(n_model))

    model = []
    weights = []
    kmers = []
    i = 0
    for filename in os.listdir('.'):
        if fnmatch.fnmatch(filename, model_filename_pattern):
            i = i + 1
            print(time_string(),
                  'loading ' + str(i) + ' of ' + str(n_model) + ' models: ' + filename + '                      ',
                  end='\r')
            sys.stdout.flush()
            K.clear_session()
            model = load_model(filename, compile=compile_model)
            weights.append(model.get_weights())
    print("")

    print(time_string(), "calculating info and rank for " + str(n_model) + " models                    ", end='\r')
    infos = numpy.zeros((n_model, n_motif))
    rnks = numpy.zeros((n_model, n_motif), dtype=int)
    topkmers = numpy.zeros((n_model, n_motif), dtype=int)
    for i in range(n_model):
        print(time_string(), "calculating info and rank for " + str(n_model) + " models: " + str(i + 1), end='\r')
        sys.stdout.flush()
        info, rnk, topkmer, kmers = information_content(weights[i][:2], motifs, kmers)
        infos[i, :] = info
        rnks[i, :] = rnk
        topkmers[i, :] = topkmer
    numpy.savetxt('info.txt', infos, fmt='%.4f')
    numpy.savetxt('rnks.txt', rnks, fmt='%d')
    numpy.savetxt('topkmers.txt', rnks, fmt='%d')

    print(time_string(), "merge model")
    weight, infos, topkmers = merge_models(weights, method, power, min_info, infos, topkmers)
    merged_info, merged_rnk, topkmer, kmers = information_content(weight[:2], motifs, kmers)
    print(merged_rnk)
    print(merged_info[merged_rnk == 1])

    model.set_weights(weight)
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.save('motif' + str(n_motif) + '-merge-' + str(n_model) + '-' + method + '-' + str(power) + '.best_model.h5')
    first_rnk_count = numpy.sum(rnks == 1, axis=0)
    print(sum(first_rnk_count > 0))
    print(sum(merged_rnk == 1))
    return model, weights, motifs, infos, rnks, topkmers, merged_info, merged_rnk, first_rnk_count


def saturation(weights, method, power, min_info, infos, topkmers, motifs):
    kmers = []
    correct = [[0]] * len(weights)
    for i in range(100, len(weights), 10):
        weight, infos, topkmers = merge_models(weights[:i + 1], method, power, min_info, infos, topkmers)
        merged_info, merged_rnk, topkmer, kmers = information_content(weight[:2], motifs, kmers)
        correct[i] = sum(merged_rnk == 1)
        print(i, correct[i])


# note that the positional effect vector is: [motif 1 region 1, motif 2 region 1, motif 3 region 1, motif 1 region 2, ...]
# test results: correlation ~ max >> average (average doesn't work well)
# need to try cor and max more to see which is better. or may be run both, train more, then merge again?
# TODO: note that some model learns an motif for the wrong RBP. To remove those models for this motif, see if it correlates better with other best motifs in other models
# how: first get a merged model. compare motifs 
# TODO: note in the returned models, models[0] is modified. try clone_model, copy weights, recompile with loss and optimizer
def merge_models(
        weights,  # a list of weights from trained models
        method,  # method to merge models: vote (default),maxinfo, correlation, average
        power=5,  # parameter used by the method
        min_info=0,  # ignore motifs with low info content
        infos=[],
        topkmers=[]
):
    n_model = len(weights)
    n_region, n_motif, l_motif = get_model_parameters_from_weights(weights[0])

    if infos == []:
        infos = numpy.zeros((n_model, n_motif))
        kmers = []
        topkmers = numpy.zeros((n_model, n_motif))
        for i in range(n_model):
            print(time_string(), "calculate information content for model " + str(i + 1), end='\r')
            sys.stdout.flush()
            info, rnk, topkmer, kmers = information_content(weights[i][:2], [], kmers)
            infos[i, :] = info
            topkmers[i, :] = topkmer

    infos[infos < min_info] = 0

    print(time_string(), "merging models                                                                         ")
    # initialize the merged model with the first model in the list. 
    merged_weights = copy.deepcopy(weights[0])

    if method == 'average':  # average weighted by info content. doesn't work well
        if power != 1:
            infos = infos ** power
        for i in range(n_motif):
            merged_weights[0][:, :, :, i] = merged_weights[0][:, :, :, i] * infos[0, i]  # motif
            merged_weights[1][i] = merged_weights[1][i] * infos[0, i]  # bias
            for x in range(n_region):  # positional effect
                merged_weights[2][i + x * n_motif] = merged_weights[2][i + x * n_motif] * infos[0, i]
            # merged_weights[2][(i*n_region):((i+1)*n_region)] =  merged_weights[2][(i*n_region):((i+1)*n_region)] * infos[0,i]   # position. this is wrong
        # positional effect bias not merged, not associated with each motif
        for i in range(1, n_model):
            for j in range(n_motif):
                merged_weights[0][:, :, :, j] = merged_weights[0][:, :, :, j] + weights[i][0][:, :, :, j] * infos[i, j]
                merged_weights[1][j] = merged_weights[1][j] + weights[i][1][j] * infos[i, j]
                for x in range(n_region):  # positional effect
                    merged_weights[2][j + x * n_motif] = merged_weights[2][j + x * n_motif] + weights[i][2][
                        j + x * n_motif] * infos[i, j]
                # merged_weights[2][(j*n_region):((j+1)*n_region)] =  merged_weights[2][(j*n_region):((j+1)*n_region)] + weights[i][2][(j*n_region):((j+1)*n_region)] * infos[i,j]
        # divide weights by total info
        total_info = numpy.sum(infos, axis=0)
        for i in range(n_motif):
            merged_weights[0][:, :, :, i] = merged_weights[0][:, :, :, i] / total_info[i]
            merged_weights[1][i] = merged_weights[1][i] / total_info[i]
            for x in range(n_region):  # positional effect
                merged_weights[2][i + x * n_motif] = merged_weights[2][i + x * n_motif] / total_info[i]
            # merged_weights[2][(i*n_region):((i+1)*n_region)] = merged_weights[2][(i*n_region):((i+1)*n_region)] / total_info[i]

    elif method == 'correlation':  # pick the one that showed high correlation with other models
        for i in range(n_motif):
            # calculate pairwise motif similarity matrix
            sim_mtx = numpy.ones((n_model, n_model))
            for j in range(n_model):
                for k in range(j + 1, n_model):
                    sim_mtx[j][k] = pwm_similarity(weights[j][0][:, :, :, i], weights[k][0][:, :, :, i])
                    sim_mtx[k][j] = sim_mtx[j][k]
            # truncate negative correlation to 0
            # sim_mtx[sim_mtx<0] = 0
            if power != 1:
                sim_mtx = sim_mtx ** power  # note that this may change sign if power is even
                infos = infos ** power
            # for each model, calculate its score and find the model with the best score
            scores = numpy.zeros(n_model)
            max_model = 0
            for j in range(n_model):
                scores[j] = numpy.sum(sim_mtx[j, :] * infos[:, i])
                if scores[max_model] < scores[j]:
                    max_model = j

            merged_weights[0][:, :, :, i] = weights[max_model][0][:, :, :, i]
            merged_weights[1][i] = weights[max_model][1][i]
            for x in range(n_region):  # positional effect
                merged_weights[2][i + x * n_motif] = weights[max_model][2][i + x * n_motif]
            # merged_weights[2][(i*n_region):((i+1)*n_region)] = weights[max_model][2][(i*n_region):((i+1)*n_region)]

    elif method == 'maxinfo':  # use model that provide max info motif
        for i in range(n_motif):
            # find the model that give the largest info
            am = numpy.argsort(infos[:, i])  # small first
            merged_weights[0][:, :, :, i] = weights[am[-1]][0][:, :, :, i]
            merged_weights[1][i] = weights[am[-1]][1][i]
            for x in range(n_region):  # positional effect
                merged_weights[2][i + x * n_motif] = weights[am[-1]][2][i + x * n_motif]
            # merged_weights[2][(i*n_region):((i+1)*n_region)] = weights[am[-1]][2][(i*n_region):((i+1)*n_region)]

    elif method == 'vote':  # find the motifs top ranked by most models. If no one, use maxinfo
        for i in range(n_motif):
            votes = numpy.zeros(n_model)
            for j in range(n_model):
                # for each model, find
                for k in range(n_model):
                    if topkmers[j, i] == topkmers[k, i]:
                        votes[j] = votes[j] + 1
            # find the model got most votes
            res = numpy.argmax(votes)
            if votes[res] == 1:
                res = numpy.argsort(infos[:, i])[-1]  # maxinfo
            else:
                # use the model with more info. what about ties of votes?
                best_model = res
                best_info = infos[res, i]
                for j in range(n_model):
                    if votes[j] == votes[res]:
                        if infos[j, i] > best_info:
                            best_model = j
                            best_info = infos[j, i]
                res = best_model
            merged_weights[0][:, :, :, i] = weights[res][0][:, :, :, i]
            merged_weights[1][i] = weights[res][1][i]
            for x in range(n_region):  # positional effect
                merged_weights[2][i + x * n_motif] = weights[res][2][i + x * n_motif]
            # merged_weights[2][(i*n_region):((i+1)*n_region)] = weights[res][2][(i*n_region):((i+1)*n_region)]

    else:
        print(time_string(), "ERROR: wrong merge method", method)
        raise

    return merged_weights, infos, topkmers


# flip a motif in a model if for 2/3 of the position, most     
def flip_inverted_motif(model):
    return model


# see updated version: merge_models    
def merge_two_models_by_motif_info(model, model2):
    # get model parameters and make sure they are the same
    n_region, n_motif, l_motif, l_seq = get_model_parameters(model)
    n_region2, n_motif2, l_motif2, l_seq2 = get_model_parameters(model2)

    if n_region != n_region2 or n_motif != n_motif2 or l_motif != l_motif2:
        print(time_string(), "ERROR: models cannot be merged!")
        raise

    # get motif information content
    kmers = []
    info, rnk, topkmer, kmers = information_content(model, [], kmers)
    info2, rnk2, topkmer, kmers = information_content(model2, [], kmers)

    motif_weights = model.layers[n_region].get_weights()
    motif_weights2 = model2.layers[n_region].get_weights()
    positional_effect = model.layers[-1].get_weights()
    positional_effect2 = model2.layers[-1].get_weights()
    for i in range(n_motif):
        if info[i] < info2[i]:
            motif_weights[0][:, :, :, i] = motif_weights2[0][:, :, :, i]  # weights
            motif_weights[1][i] = motif_weights2[1][i]  # bias
            positional_effect[0][i] = positional_effect2[0][i]
    model.layers[n_region].set_weights(motif_weights)
    model.layers[-1].set_weights(positional_effect)

    return model

    '''
    from splicenet10 import *
    model_names = ['motif20','motif20-random','motif20-3','motif20-4','motif20-2','motif20-constraint','motif20l2']
    model = load_model(model_names[0]+'.best_model.h5')
    info,rnk = information_content(model,[]) 
    print(info)
    for model_name in model_names:
        model2 = load_model(model_name+'.best_model.h5')
        info2,rnk = information_content(model2,[]) 
        model = merge_two_models_by_motif_info(model,model2)
        print(info2)
    info,rnk = information_content(model,[]) 
    model.save('motif20-merged.best_model.h5')
    
    '''


# calculate correlation coefficient between exon PSI and splicing factor expression
def calculate_exon_SF_correlation(x, y, n_experiment):
    # x and y grouped by experiment
    n_region = len(x) - 1
    n_motif = x[n_region].shape[1]
    n_exon = int(x[n_region].shape[0] / n_experiment)
    index = numpy.arange(n_experiment)
    print(n_region, n_motif, n_exon)
    # SF expression matrix, each row is an experiment, each column is a SF
    expr = x[n_region][index * n_exon, :]
    cor = numpy.zeros((n_exon, n_motif), dtype=float)
    for i in range(n_exon):
        print("calculating for exon ", i + 1, '                                     ', end='\r')
        for j in range(n_motif):
            cor[i, j] = numpy.corrcoef(y[index * n_exon + i].flatten(), expr[:, j].flatten())[0, 1]
    return cor


# use deep learning to do motif regression
# doesn't seem to work well. need more work here
################################################################

# compile and initialize a model
def motif_regressor_net(
        l_seq,  # length of sequence
        l_motif,  # length of the motif
        n_region,  # number of regions
        use_bias_motif=True,  # kernal/motif bias
        use_bias_position=True  # dense/positional effect bias
):
    individual_motif_layer = Conv2D(1, kernel_size=(4, l_motif), activation='relu', input_shape=(4, l_seq, 1),
                                    use_bias=use_bias_motif, kernel_regularizer=keras.regularizers.l2(0.0))

    total_motif_layer = AveragePooling2D(pool_size=(1, l_seq - l_motif + 1))

    flatten_layer = Flatten()

    # call the shared layers on each input sequence to compute the RBP occupancy in each region 
    all_inputs = []  # a list of input sequences from each region
    all_motif_score = []
    for i in range(n_region):
        all_inputs.append(Input(shape=(4, l_seq, 1,)))  # One input sequence layer per region
        individual_motif_score = individual_motif_layer((all_inputs[i]))
        total_motif_score = total_motif_layer(individual_motif_score)
        total_motif_score_flatten = flatten_layer(total_motif_score)
        all_motif_score.append(total_motif_score_flatten)

    # concatenate in all regions 
    all_motif_score = Concatenate(1)(all_motif_score)

    output = Dense(1,
                   kernel_initializer=keras.initializers.he_normal(seed=None),
                   use_bias=use_bias_position,
                   )(all_motif_score)

    model = Model(inputs=all_inputs, outputs=output)
    model.compile(loss='mean_squared_error', optimizer='adam')  # adam, Nadam, Adamax, RMSprop,Adadelta,sgd,Adagrad,

    return model


def motif_regressor_net_one_region(l_seq, l_motif, use_bias_motif=True, use_bias_position=True):
    individual_motif_layer = Conv2D(1, kernel_size=(4, l_motif), activation='relu', input_shape=(4, l_seq, 1),
                                    use_bias=use_bias_motif)

    total_motif_layer = AveragePooling2D(pool_size=(1, l_seq - l_motif + 1))

    flatten_layer = Flatten()

    # call the shared layers on each input sequence to compute the RBP occupancy in each region 

    all_inputs = Input(shape=(4, l_seq, 1,))  # One input sequence layer per region
    individual_motif_score = individual_motif_layer((all_inputs))
    total_motif_score = total_motif_layer(individual_motif_score)
    total_motif_score_flatten = flatten_layer(total_motif_score)
    output = Dense(1)(total_motif_score_flatten)

    model = Model(inputs=all_inputs, outputs=output)
    model.compile(loss='mean_squared_error', optimizer='adam')  # adam, Nadam, Adamax, RMSprop,Adadelta,sgd,Adagrad,

    return model


def motif_regressor_training(
        seqs_train,
        seqs_test,
        output_train,
        output_test,
        l_motif,
        batch_size,
        n_epoch,
        verbose,
        patience,
        use_bias_motif=True,
        use_bias_position=True,
):
    n_region = len(seqs_train)
    l_seq = seqs_train[0].shape[2]
    model = motif_regressor_net(l_seq, l_motif, n_region, use_bias_motif, use_bias_position)

    model.fit(
        seqs_train,
        output_train,
        batch_size=batch_size,
        epochs=n_epoch,
        verbose=verbose,
        validation_data=(seqs_test, output_test),
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                min_delta=0,
                patience=patience,
                verbose=0,
                mode='auto',
                restore_best_weights=True
            )
        ]
    )

    return model


################################################################
'''
from splicenet11 import *
model0 = load_model('motif10region4.simulator_model.h5')
x_train,y_train,x_test,y_test,seqs_train, seqs_test = splice_net_simulation(
        model0,
        [],       # RBP expression matrix, each column is an experiment, each row is a RBP. If [], will use simulation
        1000,
        100,
        1000,
        100,
        4,
        0.25,
        'experiment',       # how training data is organized: EXPERIMENT, EXON (TODO), RANDOM #TODO: systematically compare options for motif and positionaleffect learning
        False # TODO: remove PSI = 0.5 due to lack of motif match
        )
        
cor_train = calculate_exon_SF_correlation(x_train,y_train,n_expr)
cor_test = calculate_exon_SF_correlation(x_test,y_test,100)

use_bias_motif=True, 
use_bias_position=True,
n_region = len(seqs_train)
l_seq = seqs_train[0].shape[2]
l_motif=6
 

model = motif_regressor_net(l_seq,l_motif,n_region,use_bias_motif, use_bias_position)
model.fit(
            seqs_train, 
            cor_train[:,0],
            batch_size = 10,
            epochs = 30,
            verbose = 3,
            validation_data = (seqs_test, cor_test[:,0]),
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor = 'val_loss',
                    min_delta = 0,
                    patience = 3,
                    verbose = 0, 
                    mode = 'auto',
                    restore_best_weights=True
                    )
                ]
            ) 
numpy.transpose(model.layers[4].get_weights()[0][:,:,:,0][:,:,0])

numpy.savetxt('pwm.txt',numpy.transpose(model.layers[4].get_weights()[0][:,:,:,0][:,:,0]),fmt='%.3f')            

'''


def parse_pwm_from_matrix_reduce_output(filename):
    # filename: path to a psam file in MatrixREDUCE output folder, typically psam_001.xml

    cmd = "more " + filename + "  | grep '#' | grep -v '=' | grep -v opt | cut -d '#' -f 1 >" + filename + "tmp"
    os.system(cmd)
    pwm = numpy.loadtxt(filename + 'tmp')
    os.system('rm ' + filename + 'tmp')

    return numpy.transpose(pwm)


def parse_positional_effect_from_matrix_reduce_log(filename):
    # filename: path to MatrixREDUCE.log file

    cmd = 'more '+ filename + ' | grep "statistics" -A 3 | grep slope | head -n 1 | cut -f 2 | sed "s/=/\t/g" | cut -f 2 >' + filename + '.statistics '
    os.system(cmd)
    positional_effect = numpy.loadtxt(filename+'.statistics')

    return positional_effect

def positional_effect_normalization(pe,scale):
    # scale positional effect array such that the sum is close to 0
    pe[pe>0] = pe[pe>0] / sum(pe[pe>0])
    pe[pe < 0] = -pe[pe < 0] / sum(pe[pe < 0])
    pe = pe * scale / (max(pe)-min(pe)) * 2
    return pe
