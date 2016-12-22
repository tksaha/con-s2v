#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os 
import sys
import pickle
import gensim 
import numpy as np

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU
from keras.datasets import imdb
from sklearn.preprocessing import LabelEncoder
from utility.Utility import Utility
from baselineRunner.SupervisedBaselineRunner import SupervisedBaselineRunner


class RNNLSTMRunner (SupervisedBaselineRunner):
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


    def _getConfusionMatrix(self):
        return mt.confusion_matrix(self.true_values, self.predicted_values, labels = self.class_keys)
        
    def _getCohenKappaScore(self):
        return mt.cohen_kappa_score(self.true_values, self.predicted_values, labels = self.class_keys)
        
    def _getClassificationReport(self):
        return mt.classification_report(self.true_values, self.predicted_values,\
             labels = self.class_keys, target_names = self.class_names, digits=4)
    
    def _getAccuracyScore(self):
        return mt.accuracy_score(self.true_values, self.predicted_values)

    def __writeClassificationReport(self, evaluationResultFile, dummyName=""):

        evaluationResultFile.write("%s%s%s%s" %("######Classification Report",\
                    "(%s)######\n"%dummyName, \
                    self._getClassificationReport(), "\n\n"))
        evaluationResultFile.write("%s%s%s" %("######Accuracy Score######\n", \
                    self._getAccuracyScore(), "\n\n"))
        evaluationResultFile.write("%s%s%s" %("######Confusion Matrix######\n", \
                    self._getConfusionMatrix(), "\n\n"))
        evaluationResultFile.write("%s%s%s" %("######Cohen's Kappa######\n", \
                    self._getCohenKappaScore(), "\n\n"))
                    
        Logger.logr.info("Evaluation with Logistic regression Completed.")

    def prepareData(self, pd):
        pass
    def runTheBaseline(self, rbase, latent_space_size):
        pass 
    def generateSummary(self, gs,  lambda_val=1.0, diversity=False):
        pass

    def countFreq (self, sentence):
        content = gensim.utils.to_unicode(sentence) 
        content = self.utFunction.normalizeText(content, remove_stopwords=0)

        for token in content:
            self.word_counter[token] += 1

    def runLSTMBaseline(self, rbase, latent_space_size):

        Logger.logr.info ("Running LSTM (RNN) with following"\
            " configuration: batch_size = %i "\
            " nb_filter = %i "\
            " filter_length = %i "\
            " percent vocab size = %i "\
            " nb_epoch = %i "%(self.batch_size, self.nb_filter,\
                 self.filter_length, self.percent_vocab_size, \
                 self.nb_epoch))

        model = Sequential()
		model.add(Embedding(self.max_features, 128, dropout=self.dropout))


        # We Could use simpleRNN or GRU instead
		model.add(LSTM(128, dropout_W=self.dropout_W, dropout_U=self.dropout_U)) 
		model.add(Dense(self.n_classes))
		model.add(Activation(self.activation_out))


    def getlistOfIndexes(self, sentence):
        content = gensim.utils.to_unicode(sentence) 
        content = self.utFunction.normalizeText(content, remove_stopwords=0)

        list_of_indexes = []
        for token in content:
            if token in self.filtered_words:
                list_of_indexes.append(self.filtered_words[token])

        #Logger.logr.info(list_of_indexes)
        return list_of_indexes



    def doHouseKeeping(self):
        self.postgresConnection.disconnectDatabase()