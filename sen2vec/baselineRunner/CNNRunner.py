#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os 
import sys
import pickle
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Convolution1D, GlobalMaxPooling1D
from keras.datasets import imdb
from keras import backend as K
from log_manager.log_config import Logger 



class CNNRunner (BaselineRunner):
    def __init__(self, *args, **kwargs):
        """
        """
        BaselineRunner.__init__(self, *args, **kwargs)
        self.postgresConnection.connectDatabase()
        self.percent_vocab_size = 80
        self.maxlen = 400
        self.dropout = 0.2
        self.nb_filter = 250
        self.filter_length = 3
        self.border_mode = 'valid'
        self.activation_h  = 'relu'
        self.activation_out = 'sigmoid'
        self.subsample_length = 1
        self.hidden_dims = 250
        self.optimizer = 'adam'
        self.loss = 'categorical_crossentropy'
        self.metric_list = ['accuracy', 'precision', 'recall', 'fbeta_score']
        self.nb_epoch = 2
        self.batch_size = 64 
        self.model = None 


    def prepareData(self, pd):
        pass

    def runTheBaseline(self, rbase, latent_space_size):

        Logger.logr.info ("Runnin CNN with following"\
            " configuration: batch_size = %i "\
            " nb_filter = %i "\
            " filter_length = %i "\
            " percent vocab size = %i "\
            " nb_epoch = %i "%(self.batch_size, self.nb_filter,\
                 self.filter_length, self.percent_vocab_size, \
                 self.nb_epoch))

        self.model = Sequential()

        # We start off with an efficient embedding layer which maps
        # our vocab indices into embedding_dims dimensions
        self.model.add(Embedding(self.max_features,
                    latent_space_size*2,
                    input_length=self.maxlen,
                    dropout=self.dropout))

        # We add a Convolution1D, which will learn nb_filter
        # word group filters of size filter_length:
        self.model.add(Convolution1D(nb_filter = self.nb_filter,
                        filter_length = self.filter_length,
                        border_mode = self.border_mode, 
                        activation = self.activation
                        subsample_length = self.subsample_length))
        # We use max pooling:
        self.model.add(GlobalMaxPooling1D())

        # We add a vanilla hidden layer:
        self.model.add(Dense(self.hidden_dims))
        self.model.add(Dropout(self.dropout))
        self.model.add(Activation(self.activation_h))

        # We project onto a single unit output layer, 
        # and squash it with a sigmoid:
        self.model.add(Dense(1))
        self.model.add(Activation(self.activation_out))

        self.model.compile(loss=self.loss,
              optimizer = self.optimizer, 
              metrics=self.metric_list)



    def generateSummary(self, gs,  lambda_val=1.0, diversity=False):
        pass

    def runEvaluationTask(self,  rbase, latent_space_size):
        # Run the cnn
        for self.batch_size in [16, 32, 64, 128]:
            for self.nb_filter in [150, 200, 250, 300]:
                for self.filter_length in [2, 3, 4]:
                    for self.percent_vocab_size in [80, 85, 90, 95]:
                        tr_X, tr_Y, ts_X, ts_Y, val_X, val_Y = getData(self.percent_vocab_size)
                        for self.nb_epoch in [2, 5, 7]:
                            self.model = self.runTheBaseline (1, latent_space_size)


    def doHouseKeeping(self):
        self.postgresConnection.disconnectDatabase()
        
