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
from subprocess import Popen 



from baselineRunner.BaselineRunner import BaselineRunner

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
        self.metric_list = ['accuracy']
        self.nb_epoch = 2
        self.batch_size = 64 
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

    def runCNNBaseline(self, rbase):
        """
        We start off with an efficient embedding layer which maps
        our vocab indices into embedding_dims dimensions. 
        We add a Convolution1D, which will learn nb_filter word 
        group filters of size filter_length. We use max pooling:
        """
        Logger.logr.info ("Running CNN with following"\
            " configuration: batch_size = %i "\
            " nb_filter = %i "\
            " filter_length = %i "\
            " percent vocab size = %i "\
            " nb_epoch = %i "%(self.batch_size, self.nb_filter,\
                 self.filter_length, self.percent_vocab_size, \
                 self.nb_epoch))

        self.model = Sequential()
        self.model.add(Embedding(self.max_features, 50, input_length=self.maxlen,
                    dropout=self.dropout))
        self.model.add(Convolution1D(nb_filter = self.nb_filter,
                        filter_length = self.filter_length,
                        border_mode = self.border_mode, 
                        activation = self.activation_h,
                        subsample_length = self.subsample_length))
        
        self.model.add(GlobalMaxPooling1D())
        self.model.add(Dense(self.hidden_dims))
        self.model.add(Dropout(self.dropout))
        self.model.add(Activation(self.activation_h))
        self.model.add(Dense(self.n_classes))
        self.model.add(Activation(self.activation_out))
        self.model.compile(loss=self.loss, optimizer = self.optimizer,  metrics=self.metric_list)

    def getlistOfIndexes(self, sentence):
        content = gensim.utils.to_unicode(sentence) 
        content = self.utFunction.normalizeText(content, remove_stopwords=0)

        list_of_indexes = []
        for token in content:
            if token in self.filtered_words:
                list_of_indexes.append(self.filtered_words[token])

        #Logger.logr.info(list_of_indexes)
        return list_of_indexes

    def encodeLabel (self, Y):
        if  self.isfirstTimeEncoding:
            self.encoder.fit(Y)
            self.isfirstTimeEncoding = False 
        
        encoded_Y = self.encoder.transform(Y)
        return encoded_Y

    def getData (self, percent_vocab_size):
        # Leave id zero out of index, it is used for padding 
        summaryMethodID = 2
        train_sentences, train_labels = [], []
        test_sentences, test_labels  = [], []
        valid_sentences, valid_labels = [], []

    
        for result in self.postgresConnection.memoryEfficientSelect(["sentence.id", "content", "sentence.topic"],\
             ["sentence,summary"], [["sentence.id", "=", "summary.sentence_id"],\
                ["summary.method_id", "=", summaryMethodID], ['sentence.istrain','=',"'YES'"]\
                 ], [], []):
                for row_id in range(0, len(result)):
                    train_sentences.append (result[row_id][1])
                    train_labels.append (result [row_id][2])
                    self.countFreq(result[row_id][1])
                    

        #Logger.logr.info (len(train_sentences))
        for result in self.postgresConnection.memoryEfficientSelect(["sentence.id","content", "sentence.topic"],\
             ["sentence,summary"], [["sentence.id", "=", "summary.sentence_id"],\
                ["summary.method_id", "=", summaryMethodID], ['sentence.istrain','=',"'NO'"] ], [], []):
                for row_id in range(0, len(result)):
                    test_sentences.append(result[row_id][1])
                    test_labels.append(result[row_id][2])
                    self.countFreq(result[row_id][1])

        #Logger.logr.info (len(train_sentences))  
        for result in self.postgresConnection.memoryEfficientSelect(["sentence.id", "content", "sentence.topic"],\
             ["sentence,summary"], [["sentence.id", "=", "summary.sentence_id"],\
                ["summary.method_id", "=", summaryMethodID], ['sentence.istrain','=',"'VALID'"] ], [], []):
                for row_id in range(0, len(result)):
                    valid_sentences.append(result[row_id][1])
                    valid_labels.append(result[row_id][2])
                    self.countFreq(result[row_id][1])

        #Logger.logr.info (len(train_sentences)) 
        total_word = len(self.word_counter)
        total_to_take = int(total_word * percent_vocab_size) - 1 # minus 1 for keep first place for padding


        self.filtered_words = {}
        id_ = 1
        for word, count in self.word_counter.most_common(total_to_take):
            self.filtered_words[word] = id_
            id_ = id_ + 1

        # prepare data 
        self.tr_x = []
        for sentence in train_sentences:
            self.tr_x.append(self.getlistOfIndexes(sentence))

        self.ts_x = []
        for sentence in test_sentences:
            self.ts_x.append(self.getlistOfIndexes(sentence))

        self.val_x = []
        for sentence in valid_sentences: 
            self.val_x.append(self.getlistOfIndexes(sentence))

        self.tr_x = sequence.pad_sequences(self.tr_x, maxlen=self.maxlen)
        self.ts_x = sequence.pad_sequences(self.ts_x, maxlen=self.maxlen)
        self.val_x = sequence.pad_sequences(self.val_x, maxlen=self.maxlen)

        self.n_classes = len(np.unique(train_labels))

        self.tr_y  = np_utils.to_categorical(self.encodeLabel(train_labels), self.n_classes)
        self.val_y_prime = np_utils.to_categorical(self.encodeLabel(valid_labels), self.n_classes)
        self.ts_y  = self.encodeLabel(test_labels)
        self.val_y  = self.encodeLabel(valid_labels)

        self.max_features = total_to_take
        Logger.logr.info ("Total Number of Classes = %i" %self.n_classes)

        

    def run (self):
        self.runCNNBaseline (1)
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
        import gc 
        for self.batch_size in [16]:
            for self.nb_filter in [50]:
                for self.filter_length in [2]:
                    for self.percent_vocab_size in [80]:
                        self.getData(self.percent_vocab_size)
                        for self.nb_epoch in [2]:
                            self.run ()
                            metric[(self.batch_size, self.nb_filter,\
                            self.filter_length, self.percent_vocab_size,\
                             self.nb_epoch)] = self.metric_val 
                            Logger.logr.info ("F1 value =%.4f"%self.metric_val)
                            gc.collect()

        (self.batch_size, self.nb_filter, self.filter_length, self.percent_vocab_size,\
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
        result['predicted_values'] = self.model.predict_classes(self.ts_x)
        result['true_values'] = self.ts_y

        labels = set(result['true_values'])
        class_labels = {}
        for i, label in enumerate(labels):
            class_labels[label] = label
            
        self.true_values =  result['true_values']
        self.predicted_values = result['predicted_values']
        self.class_keys = sorted(class_labels)
        self.class_names = [class_labels[key] for key in self.class_keys]

        evaluationResultFile = open("%s/%seval_%i.txt"%(self.trainTestFolder,\
                latReprName, summaryMethodID), "w")
        self.__writeClassificationReport(evaluationResultFile, latReprName)



    def doHouseKeeping(self):
        self.postgresConnection.disconnectDatabase()