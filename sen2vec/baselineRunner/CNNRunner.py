#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os 
import sys
import pickle
import gensim 
import numpy as np
import pandas as pd 
from sklearn import linear_model
import sklearn.metrics as mt
from collections import Counter 
from keras.utils import np_utils
from keras import backend as K
from log_manager.log_config import Logger 
from keras.layers import Embedding
from keras.models import Sequential
from multiprocessing import Process
from keras.preprocessing import sequence
from utility.Utility import Utility
from sklearn.preprocessing import LabelEncoder
from keras.layers import Dense, Dropout, Activation
from keras.layers import Convolution1D, GlobalMaxPooling1D
from baselineRunner.SupervisedBaselineRunner import SupervisedBaselineRunner
from keras.callbacks import EarlyStopping, ModelCheckpoint




class CNNRunner (SupervisedBaselineRunner):
    def __init__(self, *args, **kwargs):
        """
        """
        SupervisedBaselineRunner.__init__(self, *args, **kwargs)
        self.postgresConnection.connectDatabase()
        self.percent_vocab_size = 80

        # Validation over 100, 200, 300
        self.maxlen = 100

        # validation on dropout
        self.dropout = 0.2
        self.nb_filter = 250
        self.filter_length = 3
        self.border_mode = 'valid'
        self.activation_h  = 'relu'
        self.activation_out = 'softmax'
        self.subsample_length = 1
        self.hidden_dims = 128
        self.embedding_dims = 100
        self.optimizer = 'rmsprop'
        self.loss = 'categorical_crossentropy'
        self.metric_list = ['accuracy']
        self.nb_epoch = 50
        self.batch_size = 64 
        self.model = None 
        self.trainTestFolder = os.environ['TRTESTFOLDER']

        self.utFunction = Utility("Text Utility")
        self.model_filepath = os.path.join(self.trainTestFolder, "cnn_weights.hdf5");
        self.early_stopping = EarlyStopping(monitor='val_loss', patience=5)
        self.checkpointer = ModelCheckpoint(filepath=self.model_filepath, monitor='val_loss', verbose=1, save_best_only=True)
        self.true_values, self.predicted_values, self.class_keys, self.class_names = {}, {}, {}, {}
        self.n_classes  = 1
        self.encoder = LabelEncoder()
        self.isfirstTimeEncoding = True 
        self.word_counter = Counter() 


        np.random.seed(2016)

        self.max_features = None 
        self.tr_x,  self.tr_y, self.ts_x, self.ts_y = None, None, None, None  
        self.val_x, self.val_y, self.val_y_prime, self.metric_val = None, None, None, None 
        
        self.latReprName = "cnn"
    
    def prepareData(self, pd):
        pass

    def runTheBaseline(self, rbase, latent_space_size):
        pass 
    def generateSummary(self, gs,  lambda_val=1.0, diversity=False):
        pass

    def runCNNBaseline(self, rbase):
        """
        We start off with an efficient embedding layer which maps
        our vocab indices into embedding_dims dimensions. 
        We add a Convolution1D, which will learn nb_filter word 
        group filters of size filter_length. We use max pooling:
        """
        Logger.logr.info ("Running CNN with the following "\
            " configuration: batch_size = %i "\
            " maxlen = %i "\
            " embedding dim = %i "\
            " nb_filter = %i "\
            " filter_length = %i "\
            " dropout = %0.2f "\
            " percent vocab size = %i "\
            " nb_epoch = %i "%(self.batch_size, self.maxlen, self.embedding_dims, self.nb_filter,\
                 self.filter_length, self.dropout, self.percent_vocab_size, \
                 self.nb_epoch))


        self.model = Sequential()
        self.model.add(Embedding(self.max_features, self.embedding_dims, input_length=self.maxlen,
                    dropout=self.dropout))
        self.model.add(Convolution1D(nb_filter = self.nb_filter,
                        filter_length = self.filter_length,
                        border_mode = self.border_mode, 
                        activation = self.activation_h,
                        subsample_length = self.subsample_length))
        
        self.model.add(GlobalMaxPooling1D())
        # The model parameter is becoming huge with this as hidden dimension
        self.model.add(Dense(self.hidden_dims)) 
        self.model.add(Dropout(self.dropout))
        self.model.add(Activation(self.activation_h))
        self.model.add(Dense(self.n_classes))
        self.model.add(Activation(self.activation_out))
        self.model.compile(loss=self.loss, optimizer = self.optimizer,  metrics=self.metric_list)

    def run (self):

        self.runCNNBaseline (1)
        self.checkpointer = ModelCheckpoint(filepath=self.model_filepath, monitor='val_loss', verbose=1, save_best_only=True)
        self.model.fit(self.tr_x,  self.tr_y, batch_size=self.batch_size,\
             nb_epoch=self.nb_epoch, shuffle=True,\
             validation_data= (self.val_x, self.val_y_prime), callbacks=[self.early_stopping, self.checkpointer])
        self.model.load_weights(self.model_filepath)
        result = pd.DataFrame()
        result['predicted_values'] = self.model.predict_classes(self.val_x, batch_size=64)
        result['true_values'] = self.val_y 

        self.metric_val = mt.f1_score(result['true_values'],\
                result['predicted_values'], average = 'macro') 

    def runEvaluationTask(self,  rbase, latent_space_size):
        # Run the cnn validation 
        metric = {}
        summaryMethodID = 2
       

        import gc 
        for self.batch_size in [16]:
            for self.maxlen in [100, 200]:
                for self.embedding_dims in [64, 128]:
                    for self.nb_filter in [200, 300]:
                        for self.filter_length in [3]:
                            for self.dropout in [0.2, 0.3]:
                                for self.percent_vocab_size in [50, 60]:
                                    self.getData(self.percent_vocab_size)
                                    for self.nb_epoch in [20]:
                                        self.run ()
                                        metric[(self.batch_size, self.maxlen, self.embedding_dims,\
                                            self.nb_filter, self.filter_length, self.dropout, self.percent_vocab_size,\
                                         self.nb_epoch)] = self.metric_val 
                                        Logger.logr.info ("F1 value =%.4f"%self.metric_val)
                                        gc.collect()

        (self.batch_size, self.maxlen, self.embedding_dims,\
        self.nb_filter, self.filter_length, self.dropout, self.percent_vocab_size,\
        self.nb_epoch) = max(metric, key=metric.get)

        Logger.logr.info ("Optimal "\
            " configuration: batch_size = %i "\
            " maxlen = %i "\
            " embedding dim = %i "\
            " nb_filter = %i "\
            " filter_length = %i "\
            " dropout = %0.2f "\
            " percent vocab size = %i "\
            " nb_epoch = %i "%(self.batch_size, self.maxlen, self.embedding_dims, self.nb_filter,\
                 self.filter_length, self.dropout, self.percent_vocab_size, \
                 self.nb_epoch))

    

        self.getData(self.percent_vocab_size)
        self.runCNNBaseline (1)
        self.model.fit(self.tr_x,  self.tr_y, batch_size=self.batch_size,\
             nb_epoch=self.nb_epoch, shuffle=True,\
             validation_data= (self.val_x, self.val_y_prime), callbacks=[self.early_stopping, self.checkpointer])
        self.model.load_weights(self.model_filepath)
        result = pd.DataFrame()
        result['predicted_values'] = self.encoder.inverse_transform(self.model.predict_classes(self.ts_x))
        result['true_values'] = self.encoder.inverse_transform(self.ts_y)

        labels = set(result['true_values'])
        class_labels = {}
        for i, label in enumerate(labels):
            class_labels[label] = label
            
        self.true_values =  result['true_values']
        self.predicted_values = result['predicted_values']
        self.class_keys = sorted(class_labels)
        self.class_names = [class_labels[key] for key in self.class_keys]

        evaluationResultFile = open("%s/%seval_%i.txt"%(self.trainTestFolder,\
                self.latReprName, summaryMethodID), "w")
        Logger.logr.info(evaluationResultFile)
        self._writeClassificationReport(evaluationResultFile, self.latReprName)



    def doHouseKeeping(self):
        self.postgresConnection.disconnectDatabase()