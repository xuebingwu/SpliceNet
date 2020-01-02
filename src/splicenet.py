from __future__ import print_function

import os, sys, copy, fnmatch

# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = "";

# turn off warning and info
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from optparse import OptionParser

import keras
from keras.models import Sequential, Model, load_model, clone_model
from keras.layers import Dense, Dropout, Flatten, Add, Multiply, Input, Conv2D, AveragePooling2D, Concatenate

from utils import *


import numpy
import pickle

from keras.utils import multi_gpu_model

# example useage:
# python splicenet.py  --matrix_reduce --n_motif=100 --n_region=4 --motif_degeneracy 0.99 --job_name=motif100region4degeneracy0.99 --effect_scale=2000 --n_mismatch 2 --n_exon_train=2000 --n_experiment_train=2000 --RBP_expr=../data/RBP_GTEx_top1000_mean_normalized.txt --psi_noise 0.1

#TODO: try real sequence real psi/expr, see if MatrixREDUCE on subset of samples learn different motifs

# TODO need to implement multi_gpu_model. It changes layers. Need to re-wrote get_parameters etc.
#TODO: automatically adjust effect_scale until PSI distribution looks good

# create and compile a splice net model
def splice_net_model(
        n_motif,  # number of motif = number of RBP
        n_region,  # number of regions
        l_seq,  # input sequence length
        l_motif,  # motif length
        output_activation,  # '', sigmoid, relu, tanh, etc
        use_constraints,
        # kernal/bias contraints to enforce pwm-like weight matrix in convolutional filter layer: not working well
        motif_combination,  # if true, shuffle connection between motif and RBP: haven't tested
        optimizer,  # adam, rmsprop, ...
        l2_regularizer  # the scale parameter, such as 0.01.
):
    # Below are the 4 shared layers used to calcualte the RBP occupancy in each region.
    # By using shared layers, each RBP will be associated with a single motif in all regions

    # 1. a convolutional layer with each filter corresponding to a motif    

    if use_constraints:
        # TODO: how to properly enforce constraints to make the filter weight matrix like motif PWM.
        # TODO: try SeparableConv2D
        # The constraints commented out below do not work well for n_motif>10 
        individual_motif_layer = Conv2D(n_motif,
                                        kernel_size=(4, l_motif),
                                        activation='relu',
                                        input_shape=(4, l_seq, 1),
                                        # kernel_initializer = keras.initializers.he_normal(seed = None),
                                        # kernel_initializer = keras.initializers.RandomUniform(minval=0.0, maxval=1.0, seed=None),
                                        kernel_constraint=keras.constraints.NonNeg(),
                                        # kernel_constraint = keras.constraints.MinMaxNorm(min_value = 0.0, max_value = 1.0, rate = 1.0, axis = 0),
                                        # kernel_regularizer = sparsity_regularizer,#keras.regularizers.l2(0.01),
                                        kernel_regularizer=keras.regularizers.l2(l2_regularizer),
                                        # bias_initializer = keras.initializers.RandomUniform(minval=-2.0, maxval=0, seed=None)
                                        )
    else:  # default
        individual_motif_layer = Conv2D(
            n_motif,
            kernel_size=(4, l_motif),
            activation='relu',  #
            # activation = keras.layers.LeakyReLU(alpha=0.1),# TODO: test leakyReLU systematically
            input_shape=(4, l_seq, 1),
            kernel_initializer=keras.initializers.he_normal(seed=None),  # seems to be better than defaults initializer
            kernel_regularizer=keras.regularizers.l2(l2_regularizer)
        )

    # 2. Average pooling layer: sum over motif scores in each region
    # TODO: test averagepooling and maxpooling for motif identificaiton. maxpooling may help prevent learning inverted motif pwm?
    # TODO: the problem with averagepooling is that will be dominated by weak motif match if bias is positive? however seems can only learn 1 motif
    # TODO: may be don't use bias in CONV2d, but then set a threshold in relu
    total_motif_layer = AveragePooling2D(pool_size=(1, l_seq - l_motif + 1))
    # TODO: note that AAA will be counted 3 times in the context of AAAAA. To count less, use maxpooling of size 2 or 3 before averagepooling. this has low priority

    # 3. flatten the output  
    flatten_layer = Flatten()

    if motif_combination:
        if use_constraints:
            motif_combination_layer = Dense(n_motif,
                                            kernel_constraint=keras.constraints.NonNeg(),
                                            kernel_regularizer=keras.regularizers.l2(l2_regularizer)
                                            )
        else:
            motif_combination_layer = Dense(n_motif, kernel_regularizer=keras.regularizers.l2(l2_regularizer))

    # 4. Multiply the total motif score in each region to each RBP abundance
    motif_RBP_pairing_layer = Multiply()

    # call the shared layers on each input sequence to compute the RBP occupancy in each region 
    all_inputs = []  # a list of input sequences from each region
    all_RBP_occupancy = []
    input_RBP_expr = Input(shape=(n_motif,))  # RBP abundance input layer
    for i in range(n_region):
        all_inputs.append(Input(shape=(4, l_seq, 1,)))  # One input sequence layer per region
        individual_motif_score = individual_motif_layer((all_inputs[i]))
        total_motif_score = total_motif_layer(individual_motif_score)
        total_motif_score_flatten = flatten_layer(total_motif_score)
        if motif_combination:
            motif_combination_score = motif_combination_layer(total_motif_score_flatten)
            RBP_occupancy = motif_RBP_pairing_layer([motif_combination_score, input_RBP_expr])
        else:
            RBP_occupancy = motif_RBP_pairing_layer([total_motif_score_flatten, input_RBP_expr])
        all_RBP_occupancy.append(RBP_occupancy)

    # concatenate RBP occupancy in all regions 
    all_RBP_occupancy = Concatenate(1)(all_RBP_occupancy)

    # output exon splicing level: PSI
    if output_activation == '':
        PSI = Dense(1,
                    kernel_initializer=keras.initializers.he_normal(seed=None),
                    # kernel_initializer = keras.initializers.RandomUniform(minval=-1, maxval=1, seed=None),
                    # kernel_regularizer = keras.regularizers.l2(0.001),
                    )(all_RBP_occupancy)
    else:
        PSI = Dense(1,
                    activation=output_activation,
                    kernel_initializer=keras.initializers.he_normal(seed=None),
                    # kernel_initializer = keras.initializers.RandomUniform(minval=-1, maxval=1, seed=None),
                    # kernel_regularizer = keras.regularizers.l2(0.001),
                    )(all_RBP_occupancy)

    all_inputs.append(input_RBP_expr)

    model = Model(inputs=all_inputs, outputs=PSI)

    # model  = multi_gpu_model(model,gpus=4)

    model.compile(loss='mean_squared_error', optimizer=optimizer)  # adam, Nadam, Adamax, RMSprop,Adadelta,sgd,Adagrad,

    return model


# initialize the splice net with random motif + positional effect, or copy from a pre-defined model
def initialize_splice_net(
        model,  # a new model
        mode,  # copy_motif, copy_positional_effect, copy_both, or simulation
        predefined_model,  # a predefined model, if [], will initialize with random motif and positional effect
        effect_scale,  # RBP positional effect scaling factors
        fraction_functional,  # fraction of the regions that are functional
        n_mismatch,  # number of mismatch allowed
        motif_degeneracy   #
):
    if mode == 'keras_default':
        # do not change default values
        return model, 'motif', 1.0

    # get model parameters
    n_region, n_motif, l_motif, l_seq = get_model_parameters(model)

    motifs = []
    positional_effect = []
    if mode == 'simulation':
        # generate random motifs, no degenerate nucleotides
        # TODO: allow degenerate nucleotides, allow load from file, allow motif of different length
        pwms = random_encoded_seqs(n_motif, l_motif)

        # convert PWM to human-friendly sequence 
        motifs = reverse_encode_seqs(pwms)

        # motif score threshold for neuron activation encoded by the bias term 
        # note that this bias will change during model training
        motif_score_threshold = l_motif - n_mismatch - 0.001  # allow mismatch

        # set the motif weight matrix
        motif_weights = model.layers[n_region].get_weights()
        for i in range(n_motif):
            pwms[i] = pwms[i] + numpy.random.uniform(0,motif_degeneracy,(4, l_motif))**8 # weights
            pwms[i] = pwm_normalization(pwms[i])
            motif_weights[0][:, :, :, i] = pwms[i].reshape(4, l_motif, 1)
            motif_weights[1][i] = - motif_score_threshold  # bias
        model.layers[n_region].set_weights(motif_weights)

        positional_effect = generate_positional_effect(n_motif, n_region, effect_scale, fraction_functional)

        w = model.layers[-1].get_weights()
        w[0] = positional_effect

        model.layers[-1].set_weights(w)

    elif mode == 'copy_motif' or mode == 'copy_both':
        # copy motif weights from another model
        if predefined_model == []:
            print(time_string(), "Error: predefiend_model is empty!")
            raise
        model.layers[n_region].set_weights(predefined_model.layers[n_region].get_weights())
    elif mode == 'copy_posiitonal_effect' or mode == 'copy_both':
        if predefined_model == []:
            print(time_string(), "Error: predefiend_model is empty!")
            raise
        model.layers[-1].set_weights(predefined_model.layers[-1].get_weights())

    return model, motifs, positional_effect


# generate training data using an initialized model
def splice_net_simulation(
        model,
        RBP_expr,  # RBP expression matrix, each column is an experiment, each row is a RBP. If [], will use simulation
        n_exon_train,
        n_exon_test,
        n_experiment_train,
        n_experiment_test,
        gamma_shape,
        gamma_scale,
        group_by,
        # how training data is organized: EXPERIMENT, EXON (TODO), RANDOM #TODO: systematically compare options for motif and positionaleffect learning
        remove_non_regulated,  # TODO: remove PSI = 0.5 due to lack of motif match
        psi_noise,   # 0-1,
        effect_scale,
        fraction_functional,
        adjust_pos_eff=True
):
    # get model parameters
    n_region, n_motif, l_motif, l_seq = get_model_parameters(model)

    # RBP expression data
    if len(RBP_expr) == 0:
        if gamma_shape > 0 and gamma_scale > 0:
            print(time_string(), "RBP expression: randomly generated from a gamma distribution")
            expression_train = numpy.random.gamma(gamma_shape, gamma_scale, size=(n_motif, n_experiment_train))
            expression_test = numpy.random.gamma(gamma_shape, gamma_scale, size=(n_motif, n_experiment_test))
        else:
            raise SystemExit('Error: need to provide a RBP matrix or gamma distribution parameters.')
    else:
        print(time_string(), "RBP expression:from GTEx")
        if n_experiment_train + n_experiment_test > RBP_expr.shape[1] or n_motif > RBP_expr.shape[0]:
            raise SystemExit('Error: not enough data from GTEx')
        expression_train = RBP_expr[:n_motif, :n_experiment_train]
        expression_test = RBP_expr[:n_motif, n_experiment_train:n_experiment_train + n_experiment_test]

    # sequence
    print(time_string(), "sequences: randomly generated")
    seqs_train = generate_input_sequences(n_region, n_exon_train, l_seq)
    seqs_test = generate_input_sequences(n_region, n_exon_test, l_seq)

    print(time_string(), "adjust effect_scale and pos_eff to get uniform distribution of PSI")
    adjustment = 1.0
    model_updated = False
    while True:
        x_test, y_test, index, input_seqs_test = generate_training_data(seqs_test, [], expression_test, model,
                                                                        gamma_shape,
                                                                        gamma_scale, n_experiment_test, group_by,
                                                                        remove_non_regulated, 0)
        f1 = sum(y_test < 0.333) / len(y_test)
        f3 = sum(y_test > 0.667) / len(y_test)
        f2 = 1-f1-f3
        if (abs(f1-0.333) < 0.05 and abs(f2-0.333) < 0.05 and abs(f3-0.333) < 0.05 ) or adjust_pos_eff == False:
            break
        elif f1 > 0.3 and f3 > 0.3 and f2 < 0.3: # V shape, reduce effect_scale by 1/3
            w = model.layers[-1].get_weights()
            w[0] = w[0] * 2/3
            model.layers[-1].set_weights(w)
            adjustment = adjustment * 2/3
            model_updated = True
        elif f2 > 0.4 and f1 < 0.3 and f3 < 0.3: # invert V shape, increase effect_scale by 1/3
            w = model.layers[-1].get_weights()
            w[0] = w[0] * 4/3
            model.layers[-1].set_weights(w)
            adjustment = adjustment *4/3
            model_updated = True
        else:
            w = model.layers[-1].get_weights()
            w[0] = generate_positional_effect(n_motif, n_region, effect_scale, fraction_functional)
            model.layers[-1].set_weights(w)
            adjustment=1.0
            model_updated = True
        print(time_string(),"psi eveness (f1,f2,f3) - effect scale adjustment: ",f1,f2,f3,adjustment,"                             ",end='\r')

    print(time_string(), "psi eveness (f1,f2,f3) - effect scale adjustment: ", f1, f2, f3, adjustment,"                           ")

    # plot PSI distribution
    plt.hist(y_test, 100)
    plt.savefig(options.job_name + '.psi-hist.png')
    plt.close()

    if psi_noise > 0:
        print(time_string(),"adding noise to test PSI",psi_noise)
        y_test2 = y_test + numpy.random.uniform(0,psi_noise,y_test.shape)
        plt.scatter(y_test.flatten(), y_test2.flatten(), s=1, alpha=0.2)
        plt.xlabel("Original")
        plt.ylabel("Noise added (" +str(psi_noise)+")")
        plt.savefig(options.job_name + '.PSI.noise.png')
        plt.close()
        y_test = y_test2

    print(time_string(), "plot motifs with updated pos_eff")
    logo_plot_for_all_motifs_in_a_model(model, options.job_name+"-true-motif",False)
    logo_plot_for_all_motifs_in_a_model(model, options.job_name+"-true-motif-normalized",True)

    print(time_string(), "save updated simulator model")
    model.save(options.job_name + '.simulator_model.h5')

    # for each RBP, how many target exons
    # for each exon, how many RBP regulators
    #targets, regulators = splice_net_summary(x_test,model)

    print(time_string(), "generate training data")
    x_train, y_train, index, input_seqs_train = generate_training_data(seqs_train, [], expression_train, model,
                                                                       gamma_shape, gamma_scale, n_experiment_train,
                                                                       group_by, remove_non_regulated, 0)
    #x_test, y_test, index, input_seqs_test = generate_training_data(seqs_test, [], expression_train, model, gamma_shape,
    #                                                                gamma_scale, n_experiment_test, group_by,
    #                                                                remove_non_regulated, 0)

    if psi_noise > 0:
        print(time_string(),"adding noise to PSI",psi_noise)
        y_train = y_train + numpy.random.uniform(0,psi_noise,y_train.shape)
        #y_test = y_test + numpy.random.uniform(0,psi_noise,y_test.shape)

    # TODO: remove non-informative data
    # sometimes due to rare motif occurrence some sequence will have no match to any motif, their PSI will be 0.5
    if remove_non_regulated:
        print(time_string(), "remove non-regulated exons")
        sel1 = numpy.where(abs(y_train - 0.5) < 0.01)
        sel2 = numpy.where(abs(y_test - 0.5) < 0.01)

        x_train2 = [[]] * (n_region + 1)
        x_test2 = [[]] * (n_region + 1)
        for j in range(n_region + 1):
            x_train2[j] = numpy.delete(x_train[j], sel1, axis=0)
            x_test2[j] = numpy.delete(x_test[j], sel2, axis=0)
        y_train2 = numpy.delete(y_train, sel1, axis=0)
        y_test2 = numpy.delete(y_test, sel2, axis=0)
        return x_train2, y_train2, x_test2, y_test2, input_seqs

    # print(x_train[-1].shape,y_train.shape,x_test[-1].shape,y_test.shape)
    return x_train, y_train, x_test, y_test, seqs_train, seqs_test


# train a model using random initializaion or parameters from another model
def splice_net_training(
        model,
        x_train,
        y_train,
        x_test,
        y_test,
        motifs,
        positional_effect,
        n_initialization,
        merge_method,  # how to merge multiple models
        output_activation,
        use_constraints,
        batch_size,
        n_epoch,
        verbose,
        patience,
        initialization_mode,
        model0,
        job_name,
        effect_scale,
        fraction_functional,
        motif_combination,
        optimizer,
        l2_regularizer,
        n_mismatch,
        motif_i  # if > 0, only train with a single motif i.
):
    # get model parameters. so far doesn't work with multi-gpu
    n_region, n_motif, l_motif, l_seq = get_model_parameters(model)

    # single motif training, need to test thoroughly
    if motif_i >= 0 and motif_i < n_motif:
        n_motif = 1
        initialization_mode = 'keras_default'
        x_train2 = x_train.copy()
        x_train2.pop(-1)
        x_test2 = x_test.copy()
        x_test2.pop(-1)
        x_train2.append(x_train[-1][:, motif_i])
        x_test2.append(x_test[-1][:, motif_i])
    else:
        # this may double memory usage?
        x_train2 = x_train
        x_test2 = x_test

    # if n_initialization > 1: train multiple models, then merge the models, and train it again.
    model_weights = []
    for i in range(n_initialization):
        # start with a new model
        new_model = splice_net_model(n_motif, n_region, l_seq, l_motif, output_activation, use_constraints,
                                     motif_combination, optimizer, l2_regularizer)

        # initialize the new model randomly, or with some parameters used in the simulation
        # fraction_functional=1
        new_model, a, b = initialize_splice_net(new_model, initialization_mode, model0, effect_scale,
                                                fraction_functional, n_mismatch,motif_degeneracy)

        # train model
        new_model.fit(
            x_train2,
            y_train,
            batch_size=batch_size,
            epochs=n_epoch,
            verbose=verbose,
            validation_data=(x_test2, y_test),
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
        new_model.save(job_name + '-model-' + str(i) + '.h5')

        loss = new_model.evaluate(x_test2, y_test, verbose=0)

        r = numpy.corrcoef(new_model.predict(x_test).flatten(), y_test.flatten())[0, 1]
        print(time_string(), "psi correlation: r^2 ", r ** 2)

        pwm_sim, pwm_dis, pos_eff_cor = model_similarity(model0, new_model)
        print(time_string(), "motif similarity", str(numpy.round(pwm_sim,2)))
        print(time_string(), "motif distance", str(pwm_dis))
        print(time_string(), "pos eff r^2", str(pos_eff_cor**2))

        model_weights.append(new_model.get_weights())

    output_model = model

    if n_initialization > 1:
        output_model.set_weights(merge_models(model_weights, merge_method)[0])
        # output_model  = multi_gpu_model(output_model,gpus=4)
        output_model.compile(loss='mean_squared_error', optimizer='adam')

    # continue to train the merged model

    if n_initialization > 1 or n_initialization == 0:
        output_model.fit(
            x_train2,
            y_train,
            batch_size=batch_size,
            epochs=n_epoch,
            verbose=verbose,
            validation_data=(x_test2, y_test),
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

    output_model.save(job_name + '.best_model.h5')
    # final prediction
    prediction = output_model.predict(x_test2)

    return output_model, prediction

def evaluate_trained_model(
        model,
        model0,
        prediction,
        y_test,
        motifs
):
    positional_effect = model0.layers[-1].get_weights()[0].flatten()

    # splicing prediction: correlation between predicted and actual splicing PSI
    r1 = numpy.corrcoef(prediction.flatten(), y_test.flatten())[0, 1]
    # splice net learning: correlation between predicted and actual positional effect 
    r2 = numpy.corrcoef(positional_effect, model.layers[-1].get_weights()[0].flatten())[0, 1]
    # fraction of positional effect predicted with the right sign
    signloss = sum(
        numpy.sign(positional_effect) == numpy.sign(model.layers[-1].get_weights()[0].flatten())) / float(
        positional_effect.shape[0])
    # calculate information content of the learned motif and ranking of the actual motif
    info, rnk, topkmer, kmers = information_content(model.get_weights()[:2], motifs)

    return r1, r2, signloss, info, rnk

# continue to train a model with new data, until no improvement can be achieved
# use the same exons
def infinite_training(
        model,
        x_test,
        y_test,
        seqs,
        model0,
        initialization_mode,
        n_initialization,
        motifs,
        positional_effect,
        effect_scale,
        RBP_expr,
        n_experiment_train,
        n_experiment_test,
        n_epoch,
        batch_size,
        verbose,
        patience,
        output_activation,
        use_constraints,
        n_exon_train,
        n_exon_test,
        gamma_shape,
        gamma_scale,
        group_by,
        remove_non_regulated,
        motif_combination,
        optimizer,
        l2_regularizer,
        n_mismatch,
        job_name):
    log = open(job_name + '.log', 'a+')

    # get model parameters
    n_region, n_motif, l_motif, l_seq = get_model_parameters(model0)

    index0 = 0
    if x_test == []:
        print(time_string(), "generate test sequences")
        seqs_test = generate_input_sequences(n_region, n_exon_test, l_seq)

        print(time_string(), "generate test data based on the sequence: ", n_experiment_test)

        x_test, y_test, index0, input_seqs_test = generate_training_data(
            seqs_test,
            [],
            RBP_expr,
            model0,
            gamma_shape,
            gamma_scale,
            n_experiment_test,
            group_by,
            remove_non_regulated,
            0
        )

        '''
        print(time_string(),"plot PSI histogram")
        plt.hist(y_test,100)
        plt.savefig(options.job_name+'.psi-hist.png')
        plt.close()
    
        print(time_string(),"save simulation test data")
        with open(job_name+'-test-data.pickle', 'wb') as f:
            pickle.dump([x_test,y_test,seqs],f)
            #TODO: only save RBP expression and sequence, save a lot of space
        '''
    if model == []:
        print(time_string(), "build a model")
        model1 = splice_net_model(n_motif, n_region, l_seq, l_motif, output_activation, use_constraints,
                                  motif_combination, optimizer, l2_regularizer)

        # initialize the new model randomly, or with some parameters used in the simulation
        fraction_functional = 1
        model, a, b = initialize_splice_net(model1, initialization_mode, model0, effect_scale, fraction_functional,
                                            n_mismatch)
        del model1

    best_loss = model.evaluate(x_test, y_test, verbose=0)
    best_model = model

    index = index0

    print(time_string(), "generate training sequences")
    seqs_train = generate_input_sequences(n_region, n_exon_train, l_seq)
    input_seqs_train = []

    '''
    print(time_string(),"generate new training data (same sequence, new expression). index =",index)
    
    x_train,y_train,index,input_seqs_train = generate_training_data(
        seqs_train,
        input_seqs_train,
        RBP_expr,
        model0,
        gamma_shape,    
        gamma_scale,
        n_experiment_train,
        group_by,
        remove_non_regulated,
        index)
        
    # initialize the model many times, train for 2 epochs, pick the best initializaiton to continue trianing
    for i in range(n_initialization):
        # start with a new model
        new_model = splice_net_model(n_motif,n_region,l_seq,l_motif,output_activation,use_constraints,motif_combination,optimizer,l2_regularizer)
        
        # initialize the new model randomly, or with some parameters used in the simulation
        fraction_functional=1
        new_model,a,b = initialize_splice_net(new_model,initialization_mode, model0,effect_scale,fraction_functional,n_mismatch)
        
        # train model for one epoch using 10% data per batch
        new_model.fit(x_train, y_train,
                      batch_size = batch_size, #int(len(y_train)/10),
                      epochs = 1, # set 1 will usually cause a nan later
                      verbose = verbose,
                      validation_data = (x_test, y_test),
                      callbacks = [keras.callbacks.EarlyStopping(monitor = 'val_loss',min_delta = 0,patience = patience,verbose = 0, mode = 'auto')])
                

        loss = new_model.evaluate(x_test, y_test, verbose=0)
        
        print(time_string(),'initialization ',i+1,'loss',loss)
        
        if loss < best_loss:
            best_loss = loss
            best_model = new_model
    
    print(time_string(),'best initialization loss',best_model.evaluate(x_test, y_test, verbose=0))
        
    model = best_model
    '''

    loss_best = 100
    loss_new = loss_best
    no_improvement = 0
    num_expr = 0

    # stop after no improvement in the last 20 try
    while (loss_new <= loss_best or no_improvement < patience):
        print(time_string(), "generate new training data. index =", index)
        x_train, y_train, index, input_seqs = generate_training_data(
            seqs_train,
            input_seqs_train,
            RBP_expr,
            model0,
            gamma_shape,
            gamma_scale,
            n_experiment_train,
            group_by,
            remove_non_regulated,
            index
        )

        print(time_string(), "fit the model, stop when no improvement in validation")
        model.fit(x_train,
                  y_train,
                  batch_size=batch_size,  # set it to 1000
                  epochs=n_epoch,  # set it to a small number like 1 or 2
                  verbose=verbose,
                  validation_data=(x_test, y_test),
                  callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0,
                                                           mode='auto')]
                  )

        loss_new = model.evaluate(x_test, y_test, verbose=0)
        if loss_new >= loss_best:
            no_improvement = no_improvement + 1
        elif loss_new < loss_best:
            loss_best = loss_new
            no_improvement = 0
            # save the best model 
            model.save(job_name + '.best_model.h5')
        num_expr = num_expr + n_experiment_train

        r = numpy.corrcoef(model.predict(x_test).flatten(), y_test.flatten())[0, 1]
        print(time_string(), "psi correlation: r^2 ",r ** 2)

        pwm_sim, pwm_dis, pos_eff_cor = model_similarity(model0, model)
        print(time_string(), "motif similarity", str(numpy.round(pwm_sim,2)))
        print(time_string(), "motif distance", str(pwm_dis))
        print(time_string(), "pos eff r^2 ", str(pos_eff_cor ** 2))

        if RBP_expr != []:
            if index + n_experiment_train > RBP_expr.shape[1]:
                break

        del x_train
        del y_train

    # load the best model
    best_model = load_model(job_name + '.best_model.h5')
    r = numpy.corrcoef(best_model.predict(x_test).flatten(), y_test.flatten())[0, 1]
    print(time_string(), "best model performance")
    print(time_string(), "psi correlation: r^2 ", r ** 2)

    pwm_sim, pwm_dis, pos_eff_cor = model_similarity(model0,best_model)
    print(time_string(), "motif similarity", str(numpy.round(pwm_sim,2)))
    print(time_string(), "motif distance", str(pwm_dis))
    print(time_string(), "pos eff r^2", str(pos_eff_cor**2))

    return best_model


#TODO: how to pick the best motif from each region run.
#TODO: may be not matrix_reduce, but other motif discovery tools. dominated by non-targeting exons?
#TODO: try meme-chip
def splicing_motif_discovery_with_MatrixREDUCE(seqs_train, x_train, y_train, n_exon, n_expr, n_motif, n_region, l_seq,
                                               l_motif):
    '''
    Use MatrixREDUCE to identify motifs and positional effect

    # python splicenet.py --matrix_reduce  --n_motif 100 --n_region 4 --load_simulator motif100region4

    # example usage using simulated data

    from splicenet import *

    model0 = load_model('motif10region4.simulator_model.h5')

    n_region,n_motif,l_motif,l_seq = get_model_parameters(model0)

    with open('motif10region4-motif.pickle','rb') as f:
        motifs,ps = pickle.load(f)

    # number of exons and experiments used for training
    n_exon=1000
    n_expr=1000

    # test: 100 exons and 100 experiments

    # simulate the data
    x_train,y_train,x_test,y_test,seqs_train, seqs_test = splice_net_simulation(model0,[],n_exon,100,n_expr,100,4,0.25,'experiment', False)

    # TODO: only extract motif and positional effect. how to get bias?
    weights, pos_eff = splicing_motif_discovery_with_MatrixREDUCE(seqs_train, x_train,y_train,n_exon,n_expr,n_motif, n_region, l_seq,l_motif)

    # check if MatrixREDUCE learns the right motif
    # rnk will be 1 if learned the correct motif
    # TODO: somehow info is very low
    info,rnk,topkmer,kmers=information_content(weights[:2],motifs)

    # set motif bias to -4. this is used by the simulator.
    weights[1] = weights[1]-4
    pos_eff[0] = positional_effect_normalization(pos_eff[0],100)

    # check if MatrixREDUCE learns positional effect. r should be close to 1
    r = numpy.corrcoef(pos_eff[0].flatten(), ps.flatten())[0, 1]

    # The following code is an example of how to use the weights learned by MatrixREDUCE
    model = splice_net_model(n_motif, n_region, l_seq, l_motif, 'relu', False, False, 'adam', 0)

    # set the model weights to those learned by MatrixREDUCE
    model.layers[n_region].set_weights(weights)
    #model.layers[n_region].set_weights(model0.layers[n_region].get_weights())

    model.layers[-1].set_weights(pos_eff)
    #model.layers[-1].set_weights(model0.layers[-1].get_weights())

    # train the model
    model.fit(
            x_train,
            y_train,
            batch_size=1000,
            epochs=1000,
            verbose=1,
            validation_data=(x_test, y_test),
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    min_delta=0,
                    patience=5,
                    verbose=0,
                    mode='auto',
                    restore_best_weights=True
                )
            ]
        )
    r1, r2, signloss, info, rnk = evaluate_trained_model(model, model0, model.predict(x_test), y_test,
                                                             ps, motifs)
    print(time_string(), r1, r2, signloss)
    print(time_string(), "motif rank", rnk)
    print(time_string(), "motif info", str(info))
    '''

    print(time_string(),"calculate the correlation between exon PSI and RBP expression on the training set")
    cor_train = calculate_exon_SF_correlation(x_train, y_train, n_expr)

    print(time_string(), "write exon-RBP correlation to files as MatrixREDUCE input")
    for i in range(n_motif):
        f1 = open('cor_PSI_SF-' + str(i) + '.txt', 'w')
        #f2 = open('cor_PSI_SF-' + str(i) + '.abs.txt', 'w')
        for j in range(n_exon):
            f1.write(str(j) + '\t' + str(cor_train[j, i]) + '\n')
            #f2.write(str(j) + '\t' + str(abs(cor_train[j, i])) + '\n')
        f1.close()
        #f2.close()

    print(time_string(), "write sequences in each region to files as MatrixREDUCE input")
    for i in range(n_region):
        seqs = reverse_encode_seqs(seqs_train[i])
        write_fasta(seqs, 'sequence-region-' + str(i) + '.fasta')

    # concatenate into a single sequence
    #os.system('cp sequence-region-0.fasta sequence-concat.fasta')
    #for i in range(1, n_region):
    #    concat_fasta(reverse_encode_seqs(seqs_train[i]), 'sequence-concat.fasta')

    # initialize a model
    model = splice_net_model(n_motif, n_region, l_seq, l_motif, 'relu', False, False, 'adam', 0)
    weights = model.layers[n_region].get_weights()
    positional_effect = model.layers[-1].get_weights()
    pvalue = [0] * n_motif * n_region

    print(time_string(), "MatrixREDUCE motif discovery in each region for each RBP")
    # TODO: may need take the majority vote from multiple regions? the positional effect is the slope
    for i in range(n_motif):
        print(time_string()," -- RBP ", i , '                                     ', end='\r')
        # strongest pos_eff across regions
        max_abs_pe = 0.0
        min_pv = 1.0
        for j in range(n_region):
            # for each region, run MatrixREDUCE and extract motif and the slope as the positional effect
            outputdir = 'MatrixREDUCE-motif-' + str(i) + '-region-' + str(j)
            os.system('rm -rf ' + outputdir)
            os.system('mkdir ' + outputdir)
            cmd = 'MatrixREDUCE -sequence=sequence-region-' + str(j) + '.fasta -meas=cor_PSI_SF-' + str(
                i) + '.txt -strand=1 -topo=X' + str(l_motif) + ' -max_motif=1 -output=' + outputdir + ' -runlog=run.log'
            os.system(cmd)

            # if a significant motif is found, i.e. psam_001.xml file exits, load the optimized pwm (psam)
            if path.exists(outputdir + '/psam_001.xml'):
                pwm = parse_pwm_from_matrix_reduce_output(outputdir + '/psam_001.xml')
            else:# otherwise parse the motif from .log file and convert it to a pwm
                pwm = parse_motif_from_matrix_reduce_log(outputdir + '/MatrixREDUCE.log')

            # extract positional effect
            positional_effect[0][i + j*n_motif][0], pvalue[i + j*n_motif] = parse_positional_effect_from_matrix_reduce_log(outputdir + '/MatrixREDUCE.log')

            # use motif/pos_eff from the region with the strongest pos_eff or min pvalue
            if abs(positional_effect[0][i + j*n_motif][0]) > max_abs_pe:
            #if pvalue[i+j*n_motif] < min_pv:
                max_abs_pe = abs(positional_effect[0][i + j*n_motif][0])
                #min_pv = pvalue[i+j*n_motif]
                weights[0][:, :, :, i][:, :, 0] = pwm

    os.system('rm -rf MatrixREDUCE')
    os.system("mkdir MatrixREDUCE")
    os.system("mv sequence*.fasta cor_* MatrixREDUCE* MatrixREDUCE")
    return weights,positional_effect


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--job_name", dest="job_name", help="job name, prefix of output files", default="job.splicenet")
    parser.add_option("--load_simulator", dest="simulator_job", help="load simulator model and data from files",
                      default="")
    parser.add_option("--load_test", dest="test_job", help="load test data from files", default="")
    parser.add_option("--load_model", dest="model_job", help="load trained model from files", default="")
    # model parameter
    parser.add_option("--n_motif", dest="n_motif", help="number of motif/RBP for simulation", type=int, default=4)
    parser.add_option("--n_motif_train", dest="n_motif_train", help="number of motif/RBP for training/prediction",
                      type=int, default=-1)
    parser.add_option("--n_region", dest="n_region",
                      help="number of regions surrounding each exon. The same RBP can have different effect on splicing when bind each region",
                      type=int, default=2)
    parser.add_option("--l_seq", dest="l_seq", help="length of input sequence, currently same for all regions",
                      type=int, default=200)
    parser.add_option("--l_motif", dest="l_motif", help="length of motif", type=int, default=6)
    parser.add_option("--n_mismatch", dest="n_mismatch", help="motif mismatch allowed in simulation", type=float,
                      default=2.0)
    parser.add_option("--n_exon_train", dest="n_exon_train", help="total number of exons for training", type=int,
                      default=1000)
    parser.add_option("--n_exon_test", dest="n_exon_test", help="total number of exons for test", type=int, default=100)
    parser.add_option("--n_experiment_train", dest="n_experiment_train", help="number of experiments (i.e. RNA-seq)",
                      type=int, default=1000)
    parser.add_option("--n_experiment_test", dest="n_experiment_test",
                      help="number of experiment for the test set in infinite training (i.e. RNA-seq)", type=int,
                      default=100)
    parser.add_option("--motif_combination", dest="motif_combination", help="shuffle motif RBP pairing during learning",
                      default=False, action='store_true')
    parser.add_option("--use_constraints", dest="use_constraints",
                      help="use PWM constraints on convolutional layer filter", default=False, action='store_true')
    parser.add_option("--output_activation", dest="output_activation",
                      help="output activation function, sigmoid or relu or tanh or none, default is sigmoid",
                      default="sigmoid")
    parser.add_option("--l2_regularizer", dest="l2_regularizer",
                      help="parameter for l2 regularizer of the motif layer. Default 0 (or 0.1/n_motif?) ", type=float,
                      default=0)
    parser.add_option("--effect_scale", dest="effect_scale", help="RBP positional effect scaling factor. Default 700 ",
                      type=float, default=700)
    parser.add_option("--motif_degeneracy", dest="motif_degeneracy", help="A noise of x^8 (x drawed from a uniform [0,motif_degeneracy] is added to motif pwm. Default 1",
                      type=float, default=0.9)
    parser.add_option("--psi_noise", dest="psi_noise", help="A noise of x (drawed from a uniform [0,psi_noise] is added to simulated PSI. Default 0",
                      type=float, default=0)
    parser.add_option("--fraction_functional", dest="fraction_functional",
                      help="Fraction of the regions that are functional. Default 1.0. ", type=float, default=1.0)
    # training
    parser.add_option("--optimizer", dest="optimizer", help="optimizer,default adam", default="adam")
    parser.add_option("--n_epoch", dest="n_epoch", help="number of epochs in training", type=int, default=-1)
    parser.add_option("--batch_size", dest="batch_size",
                      help="batch size in training. Default is 0.01% of samples (n_exon_train x n_experiment_train)",
                      type=int, default=-1)
    parser.add_option("--verbose", dest="verbose", help="verbose level during model fitting", type=int, default=1)
    parser.add_option("--patience", dest="patience", help="patience during model fitting", type=int, default=3)
    parser.add_option("--infinite_training", dest="infinite_training",
                      help="if true, will generate new data to train a model until no improvement on test data. Default n_epoch=1,batch_size=1000 for this mode. If n_initialization is set to >1, multiple inifinitely trained models will be merged",
                      default=False, action='store_true')
    parser.add_option("--matrix_reduce", dest="matrix_reduce",
                      help="if true, will use MatrixREDUCE to discover motif and positional effect, which will be used to initialize the splice net model",
                      default=False, action='store_true')
    # simulation
    parser.add_option("--RBP_expr", dest="RBP_expr",
                      help="RBP expression data, each row is an RBP and each column is an experiment (RNA-seq), if not specified, will use gamma distribution to simulate",
                      default="")
    parser.add_option("--shuffle_RBP_expr", dest="shuffle_RBP_expr", help="if true, will shuffle RBP expressions",
                      default=False, action='store_true')
    parser.add_option("--gamma_shape", dest="gamma_shape",
                      help="Gamma distribution shape parameter for RBP expression simulation", type=float, default=4)
    parser.add_option("--gamma_scale", dest="gamma_scale",
                      help="Gamma distribution scale parameter for RBP expression simulation. Default = 200/sqrt(n_motif)",
                      type=float, default=0.25)
    parser.add_option("--n_initialization", dest="n_initialization", help="number of initializations", type=int,
                      default=5)
    parser.add_option("--merge_method", dest="merge_method",
                      help="how to merge models:maxinfo,correlation,average,vote", default='maxinfo')
    parser.add_option("--merge_power", dest="merge_power", help="power parameter used in merging models", type=float,
                      default=1.0)
    parser.add_option("--initialization_mode", dest="initialization_mode",
                      help="how to initialize model during training: keras_default, copy_motif, copy_positional_effect, copy_both, simulation",
                      default="keras_default")
    parser.add_option("--group_by", dest="group_by", help="group training data by experiment, exon (TODO), or random",
                      default="experiment")
    parser.add_option("--remove_non_regulated", dest="remove_non_regulated",
                      help="if true, will remove exons without any motif match (TODO)", default=False,
                      action='store_true')
    parser.add_option("--no_plot", dest="no_plot", help="plot model diagram and output by default", default=False,
                      action='store_true')
    parser.add_option("--no_motif_logo", dest="no_motif_logo",
                      help="do not generate motif logos. Need logomaker to generate logos",
                      default=False, action='store_true')

    (options, args) = parser.parse_args()

    # if options.gamma_scale < 0:
    #    #options.gamma_scale = options.gamma_shape
    #    options.gamma_scale = 256/(options.n_motif**0.5)

    # default parameters: batch_size, l2 regularization, n_epoch    
    if options.batch_size < 1:
        if options.infinite_training:
            options.batch_size = 1000
        else:
            options.batch_size = max(1, int(options.n_exon_train * options.n_experiment_train / 10000))

    if options.l2_regularizer < 0:
        options.l2_regularizer = 0.1 / options.n_motif

    if options.n_epoch < 0:
        if options.infinite_training:
            options.n_epoch = 1
        else:
            options.n_epoch = 1000

    # save all options/parameters for this run
    log = open(options.job_name + '.log', 'a+')
    log.write(str(options) + '\n')

    print(time_string(), str(options))

    # RBP expression data
    if options.RBP_expr == '':
        RBP_expr = []
    else:
        RBP_expr = numpy.loadtxt(options.RBP_expr)
        # shuffle orders of experiments such that tissues are not clustered
        RBP_expr = RBP_expr[:, numpy.random.permutation(RBP_expr.shape[1])]
        # shuffle RBP so that each run will be different
        RBP_expr = RBP_expr[numpy.random.permutation(RBP_expr.shape[0]), :]
        # randomize RBP tissue expression
        if options.shuffle_RBP_expr:
            shuffle_RBP_expr(RBP_expr)

    # load or create a simulator model
    if options.simulator_job == "":
        print(time_string(), "create a new model")
        model0 = splice_net_model(
            options.n_motif,
            options.n_region,
            options.l_seq,
            options.l_motif,
            'sigmoid',
            options.use_constraints,
            options.motif_combination,
            options.optimizer,
            options.l2_regularizer
        )

        print(time_string(), "initialize with random motifs and positional effects")
        model0, motifs, positional_effect = initialize_splice_net(model0, "simulation", [], options.effect_scale, 1,
                                                                  options.n_mismatch,options.motif_degeneracy)

        # plot logo
        #TODO: consider plotting pos_eff along side with logo
        logo_plot_for_all_motifs_in_a_model(model0, options.job_name+"-true-motif",False)
        logo_plot_for_all_motifs_in_a_model(model0, options.job_name+"-true-motif-normalized",True)


        #TODO: note that one may need effect_scale etc to faithfully simulate the data
        print(time_string(), "save simulator model")
        model0.save(options.job_name + '.simulator_model.h5')

        #with open(options.job_name + '-motif.pickle', 'wb') as f:
        #    pickle.dump([motifs, positional_effect], f)
        #    f.close()

    else:
        print(time_string(), "load simulator model", options.simulator_job)
        model0 = load_model(options.simulator_job + '.simulator_model.h5')
        #with open(options.simulator_job + '-motif.pickle', 'rb') as f:
        #    motifs, positional_effect = pickle.load(f)

            # load test data if specificied
    seqs = []
    if options.test_job != '':
        print(time_string(), "load test data")
        with open(options.test_job + '-test-data.pickle', 'rb') as f:
            seqs, expression = pickle.load(f)
    else:
        x_test = []
        y_test = []

    # load a trained model if specified
    if options.model_job != '':
        print(time_string(), "load trained model")
        model = load_model(options.model_job + '.best_model.h5')
    else:
        model = []

    # infinite training: use the simulator to generate new training data and train the model until it no longer improves
    # allow one to train multiple models and then merge (n_initialization)
    # TODO: note for now each run will use different training sequences
    if options.infinite_training:
        weights = []
        for i in range(options.n_initialization):
            model = infinite_training(
                [],
                x_test,
                y_test,
                seqs,
                model0,
                options.initialization_mode,
                0,
                motifs,
                positional_effect,
                options.effect_scale,
                RBP_expr,
                options.n_experiment_train,
                options.n_experiment_test,
                options.n_epoch,
                options.batch_size,
                options.verbose,
                options.patience,
                options.output_activation,
                options.use_constraints,
                options.n_exon_train,
                options.n_exon_test,
                options.gamma_shape,
                options.gamma_scale,
                options.group_by,
                options.remove_non_regulated,
                options.motif_combination,
                options.optimizer,
                options.l2_regularizer,
                options.n_mismatch,
                options.job_name
            )
            model.save(options.job_name + '-model-' + str(i) + '.h5')
            weights.append(model.get_weights())

        merged_weights, infos, topkmers = merge_models(weights, 'vote', 1)
        model.set_weights(merged_weights)
        model.save(options.job_name + '-merged_model.h5')
        info, rnk, topkmer, kmers = information_content(merged_weights[:2], motifs)
        print(time_string(), "motif rank", rnk)
        print(time_string(), "motif info", str(info))
        exit()

    if options.test_job == '':
        adjust_pos_eff = False
        if options.simulator_job == "":
            adjust_pos_eff = True
        print(time_string(), "simulating training and test data using the model")
        x_train, y_train, x_test, y_test, seqs_train, seqs_test = splice_net_simulation(
            model0,
            RBP_expr,
            options.n_exon_train,
            options.n_exon_test,
            options.n_experiment_train,
            options.n_experiment_test,
            options.gamma_shape,
            options.gamma_scale,
            options.group_by,
            options.remove_non_regulated,
            options.psi_noise,
            options.effect_scale,
            options.fraction_functional,
            adjust_pos_eff
        )

        # print(time_string(),"save simulation test data")
        # with open(options.job_name+'-test-data.pickle', 'wb') as f:
        #    pickle.dump([seqs,expression],f)

    if not options.no_plot:
        # plot model structure
        keras.utils.plot_model(model0, to_file=options.job_name + '.model.png', show_shapes=True)

    # TODO only use a subset of motifs for training/prediction
    if options.n_motif_train > 0 and options.n_motif_train < options.n_motif:
        x_train[-1] = numpy.delete(x_train[-1], range(options.n_motif - options.n_motif_train), 0)
        x_test[-1] = numpy.delete(x_test[-1], range(options.n_motif - options.n_motif_train), 0)
        # change n_motif

    if model == []:
        print(time_string(),
              "train a model, or a model initialized with motifs and / or positional effect from the simulation")
        model = splice_net_model(
            options.n_motif,
            options.n_region,
            options.l_seq,
            options.l_motif,
            options.output_activation,
            options.use_constraints,
            options.motif_combination,
            options.optimizer,
            options.l2_regularizer
        )

    # MatrixREDUCE
    if options.matrix_reduce:
        options.n_initialization = 1

        weights, pos_eff = splicing_motif_discovery_with_MatrixREDUCE(seqs_train, x_train, y_train, options.n_exon_train, options.n_experiment_train,
                                                                      options.n_motif, options.n_region, options.l_seq, options.l_motif)
        # check if MatrixREDUCE learns the right motif
        # info, rnk, topkmer, kmers = information_content(weights[:2], motifs)
        # check if MatrixREDUCE learns positional effect

        r = numpy.corrcoef(pos_eff[0].flatten(), model0.layers[-1].get_weights()[0].flatten())[0, 1]

        print(time_string(), "pos_eff r^2 ", r ** 2)
        #print(time_string(), "motif rank", str(rnk))
        #print(time_string(), "motif info", str(info))

        # CNN bias. this is not that important. the model will figure it out slowly
        weights[1] = weights[1] - 4

        #TODO: how to set pos_eff_scale value. automatically determine it to fit training PSI distribution
        #TODO: seems doesn't matter much. the training process will adjust it automatically
        pos_eff_scale = 100
        pos_eff[0] = positional_effect_normalization(pos_eff[0], pos_eff_scale)

        # set the model weights to those learned by MatrixREDUCE
        model.layers[options.n_region].set_weights(weights)
        # model.layers[n_region].set_weights(model0.layers[n_region].get_weights())

        model.layers[-1].set_weights(pos_eff)
        # model.layers[-1].set_weights(model0.layers[-1].get_weights())

        print(time_string(), "evaluate the model before training")
        r = numpy.corrcoef(model.predict(x_test).flatten(), y_test.flatten())[0, 1]
        print(time_string(), "psi correlation: r^2 ", r ** 2)

        pwm_sim, pwm_dis, pos_eff_cor = model_similarity(model0, model)
        print(time_string(), "motif similarity", str(numpy.round(pwm_sim,2)))
        print(time_string(), "motif distance", str(pwm_dis))
        print(time_string(), "pos eff r^2", str(pos_eff_cor**2))

        model.fit(
            x_train,
            y_train,
            batch_size=options.batch_size,
            epochs=options.n_epoch,
            verbose=options.verbose,
            validation_data=(x_test, y_test),
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    min_delta=0,
                    patience=options.patience,
                    verbose=0,
                    mode='auto',
                    restore_best_weights=True
                )
            ]
        )

        model.save(options.job_name + '-model-MatrixREDUCE.h5')

        prediction = model.predict(x_test)

    else:
        motif_i = -1

        model, prediction = splice_net_training(
            model,
            x_train,
            y_train,
            x_test,
            y_test,
            motifs,
            positional_effect,
            options.n_initialization,
            options.merge_method,
            options.output_activation,
            options.use_constraints,
            options.batch_size,
            options.n_epoch,
            options.verbose,
            options.patience,
            options.initialization_mode,
            model0,
            options.job_name,
            options.effect_scale,
            options.fraction_functional,
            options.motif_combination,
            options.optimizer,
            options.l2_regularizer,
            options.n_mismatch,
            motif_i  # single motif
        )

        if motif_i >= 0:
            info, rnk, topkmer, kmers = information_content(model.get_weights()[:2], motifs[motif_i])

    # evaluate the model
    r = numpy.corrcoef(model.predict(x_test).flatten(), y_test.flatten())[0, 1]
    print(time_string(), "psi correlation: r^2 ", r ** 2)

    pwm_sim, pwm_dis, pos_eff_cor = model_similarity(model0,model)
    print(time_string(), "motif similarity", str(numpy.round(pwm_sim,2)))
    print(time_string(), "motif distance", str(pwm_dis))
    print(time_string(), "pos eff r^2", str(pos_eff_cor**2))

    if not options.no_plot:
        plt.scatter(y_test.flatten(), prediction.flatten(), s=1, alpha=0.3)
        plt.xlabel("Observation")
        plt.ylabel("Prediction")
        plt.savefig(options.job_name + '.scatter.png')
        plt.close()

    if not options.no_motif_logo:
        logo_plot_for_all_motifs_in_a_model(model, options.job_name+"-learned-motif",False,pwm_dis)
        logo_plot_for_all_motifs_in_a_model(model, options.job_name+"-learned-motif-normalized",True,pwm_sim)

        # generate motif logo from convolutional layers, need to have kpLogo directly callable 
        #layer1_motif(model.layers[options.n_region].get_weights(), 1000000, 0.7, 'relu', options.job_name)
        # TODO: replace the code with exact kernal2pwm transformation: https://github.com/gao-lab/kernel-to-PWM
        # TODO: initialize kernal using pwm 2 kernal transformation. but how to deal with bias

    log.close()

# TODO: batch_size may be important for convergence and choosing best motif. also try terminating each initialization training using info>0.8 as a trigger?

# TODO: for each initialization, do not over train, may be better for merging? may be save a model for each epoch

# TODO: drop the bias term? both motif and positional effect. Instead use threshold in relu for conv2d

# TODO: invert a motif matrix if at most positions, three positive and one negative, also invert bias

# TODO: write input generator to speed up training

# TODO: when providing estimated motif and positional effect, may be freeze one and update the other? do not update at the same time
# for example, freeze motif layer first if initialized with motif

# TODO: alternative framework: first RBP combination, then multiply with motif. because RBP share the same motif

# TODO: can we do 2 regions a time? i.e. freeze other regions?, with 4 region, somehow laerns an inverted motif

# TODO: for each exon, plot how it is regulated, simulation and prediction

# TODO: focus on NOVA or FOX target exons, to see if we can leran those RNA maps

# TODO: add RBP independent motif layers to learn context features, such as GC, folding etc, but how to simulate? calculate accessibility, and use for simulation

# TODO: train a CNN to predict RNA structure / accessibility, or use rnafold and use as input

# TODO: or should we learn one motif a time? or even one region one RBP, using exons showing strong correlation with the RBP? and then put them together?

# TODO: may be feasible to combine human/mouse and other mammalian RNA-seq data, if motifs/positional effects are conserved
