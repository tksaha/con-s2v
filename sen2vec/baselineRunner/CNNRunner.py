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




class CNNRunner (BaselineRunner):
    def __init__(self, *args, **kwargs):
        """
        """
        BaselineRunner.__init__(self, *args, **kwargs)
        self.postgresConnection.connectDatabase()
        self.max_features = 20000
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
        self.metric_list = 
        self.nb_epoch = 2
        self.batch_size = 64 
        self.model = None 


    def prepareData(self, pd):
        pass

    def runTheBaseline(self, rbase, latent_space_size):
        model = Sequential()

        # We start off with an efficient embedding layer which maps
        # our vocab indices into embedding_dims dimensions
        model.add(Embedding(self.max_features,
                    latent_space_size*2,
                    input_length=self.maxlen,
                    dropout=self.dropout))

        # We add a Convolution1D, which will learn nb_filter
        # word group filters of size filter_length:
        model.add(Convolution1D(nb_filter = self.nb_filter,
                        filter_length = self.filter_length,
                        border_mode = self.border_mode, 
                        activation = self.activation
                        subsample_length = self.subsample_length))
        # We use max pooling:
        model.add(GlobalMaxPooling1D())

        # We add a vanilla hidden layer:
        model.add(Dense(self.hidden_dims))
        model.add(Dropout(self.dropout))
        model.add(Activation(self.activation_h))

        # We project onto a single unit output layer, 
        # and squash it with a sigmoid:
        model.add(Dense(1))
        model.add(Activation(self.activation_out))

        model.compile(loss=self.loss,
              optimizer = self.optimizer, 
              metrics=self.metric_list)



    def generateSummary(self, gs,  lambda_val=1.0, diversity=False):
        pass

    def runEvaluationTask(self):
        pass

    def doHouseKeeping(self):
        self.postgresConnection.disconnectDatabase()
        
