#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os 
import sys
import pickle
import gensim 
import numpy as np
import pandas as pd 
from collections import Counter 
from log_manager.log_config import Logger 
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU
from keras.datasets import imdb
from sklearn.preprocessing import LabelEncoder
from utility.Utility import Utility
from baselineRunner.SupervisedBaselineRunner import SupervisedBaselineRunner


class RNNRunner (SupervisedBaselineRunner):
    def __init__(self, *args, **kwargs):
        """
        """
        SupervisedBaselineRunner.__init__(self, *args, **kwargs)
        self.postgresConnection.connectDatabase()
        self.percent_vocab_size = 80
        self.maxlen = 400
        self.dropout = 0.2
        self.dropout_W = 0.2 
        self.dropout_U = 0.2 
        self.activation_out = 'sigmoid'

        self.optimizer = 'adam'
        self.loss = 'categorical_crossentropy'
        self.metric_list = ['accuracy']

        self.nb_epoch = 2
        self.batch_size = 16

        self.model = None 
        self.utFunction = Utility("Text Utility")


        self.true_values = {}
        self.predicted_values = {}
        self.class_keys = {}
        self.class_names = {}
        self.n_classes  = 1
        self.encoder = LabelEncoder()
        self.isfirstTimeEncoding = True 
        self.word_counter = Counter() 

        np.random.seed(2016)

        self.max_features = None 
        self.tr_x = None 
        self.tr_y = None 
        self.ts_x = None 
        self.ts_y = None 
        self.val_x = None 
        self.val_y = None 
        self.val_y_prime = None 
        self.metric_val = None 
        self.trainTestFolder = os.environ['TRTESTFOLDER']
        self.latReprName = "lstm"

    def prepareData(self, pd):
        pass
    def runTheBaseline(self, rbase, latent_space_size):
        pass 
    def generateSummary(self, gs,  lambda_val=1.0, diversity=False):
        pass

    def runLSTMBaseline(self, rbase):

        Logger.logr.info ("Running LSTM (RNN) with following"\
            " configuration: batch_size = %i "\
            " percent vocab size = %i "\
            " nb_epoch = %i "%(self.batch_size, self.percent_vocab_size, \
                 self.nb_epoch))

        self.model = Sequential()
        self.model.add(Embedding(self.max_features, 50, dropout=self.dropout))


        # We Could use simpleRNN or GRU instead
        self.model.add(LSTM(128, dropout_W=self.dropout_W, dropout_U=self.dropout_U)) 
        self.model.add(Dense(self.n_classes))
        self.model.add(Activation(self.activation_out))
        self.model.compile(loss=self.loss, optimizer = self.optimizer,\
          metrics=self.metric_list)

    def run (self):
        self.runLSTMBaseline (1)
        self.model.fit(self.tr_x,  self.tr_y, batch_size=self.batch_size,\
             nb_epoch=self.nb_epoch, shuffle=True,\
             validation_data= (self.val_x, self.val_y_prime))
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
            for self.percent_vocab_size in [80]:
                self.getData(self.percent_vocab_size)
                for self.nb_epoch in [1]:
                    self.run ()
                    metric[(self.batch_size, self.percent_vocab_size,\
                     self.nb_epoch)] = self.metric_val 
                    Logger.logr.info ("F1 value =%.4f"%self.metric_val)
                    gc.collect()

        (self.batch_size, self.percent_vocab_size,\
            self.nb_epoch) = max(metric, key=metric.get)
        Logger.logr.info ("Optimal "\
            " configuration: batch_size = %i "\
            " nb_filter = %i "\
            " filter_length = %i "\
            " percent vocab size = %i "\
            " nb_epoch = %i "%(self.batch_size, self.nb_filter,\
                 self.filter_length, self.percent_vocab_size, \
                 self.nb_epoch))

    

        self.getData(self.percent_vocab_size)
        self.runCNNBaseline (1)
        self.model.fit(self.tr_x,  self.tr_y, batch_size=self.batch_size,\
             nb_epoch=self.nb_epoch, shuffle=True,\
             validation_data= (self.val_x, self.val_y_prime))
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