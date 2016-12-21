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
import sklearn.metrics as mt
import pandas as pd 
from collections import Counter 


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

        self.true_values = {}
        self.predicted_values = {}
        self.class_keys = {}
        self.class_names = {}
        self.word_counter = Counter() 

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
            word_counter[token] += 1

     def CNNBaseline(self, rbase, latent_space_size):

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

    def getlistOfIndexes(self, sentence):
        content = gensim.utils.to_unicode(sentence) 
        content = self.utFunction.normalizeText(content, remove_stopwords=0)

        list_of_indexes = []
        for token in content:
            if token in self.filtered_words:
                list_of_indexes.append(self.filtered_words[token])

        Logger.logr.info(list_of_indexes)
        return list_of_indexes

        
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
                    

        for result in self.postgresConnection.memoryEfficientSelect(["sentence.id","content", "sentence.topic"],\
             ["sentence,summary"], [["sentence.id", "=", "summary.sentence_id"],\
                ["summary.method_id", "=", summaryMethodID], ['sentence.istrain','=',"'NO'"] ], [], []):
                for row_id in range(0, len(result)):
                    test_sentences.append(result[row_id][1])
                    test_labels.append(result[row][2])
                    self.countFreq(result[row_id][1])
            

        for result in self.postgresConnection.memoryEfficientSelect(["sentence.id", "content", "sentence.topic"],\
             ["sentence,summary"], [["sentence.id", "=", "summary.sentence_id"],\
                ["summary.method_id", "=", summaryMethodID], ['sentence.istrain','=',"'VALID'"] ], [], []):
                for row_id in range(0, len(result)):
                    valid_sentences.append(result[row_id][1])
                    valid_labels.append(result[row_id][2])
                    self.countFreq(result[row_id][1])

        total_word = len(self.word_counter)
        total_to_take = int(total_word * percent_vocab_size) - 1 # minus 1 for keep first place for padding


        self.filtered_words = {}
        id_ = 1
        for word, count in self.word_counter.most_common(total_to_take):
            self.filtered_words[word] = id_
            id_ = id_ + 1

        # prepare data 
        train_x = []
        for sentence in train_sentences:
            train_x.append(self.getlistOfIndexes(sentence))

        test_X = []
        for sentence in test_sentences:
            test_x.append(self.getlistOfIndexes(sentence))

        for sentence in valid_sentences: 
            valid_x.append(self.getlistOfIndexes(sentence))

        train_x = sequence.pad_sequences(train_x, maxlen=self.maxlen)
        test_x = sequence.pad_sequences(test_x, maxlen=self.maxlen)
        val_x = sequence.pad_sequences(val_x, maxlen=self.maxlen)

        return train_x, train_labels, test_x, test_labels, val_x, valid_labels, total_to_take

    def runEvaluationTask(self,  rbase, latent_space_size):
        # Run the cnn validation 
        metric = {}


        for self.batch_size in [16, 32, 64, 128]:
            for self.nb_filter in [150, 200, 250, 300]:
                for self.filter_length in [2, 3, 4]:
                    for self.percent_vocab_size in [80, 85, 90, 95]:
                        tr_X, tr_Y, ts_X, ts_Y, val_X, val_Y, max_feat = getData(self.percent_vocab_size)
                        for self.nb_epoch in [2, 5, 7]:
                            self.max_features = max_feat
                            self.model = self.runCNNBaseline (1, latent_space_size)
                            self.model.fit(tr_X, tr_Y, batch_size=self.batch_size,\
                                 nb_epoch=self.nb_epoch, validation_data= (val_X, val_Y))
                            result = pd.DataFrame()
                            result['predicted_values'] = logit.predict(val_X)
                            result['true_values'] = val_Y

                            val = mt.f1_score(result['true_values'],\
                                    result['predicted_values'], average = 'macro') 

                            metric[(self,batch_size, self.nb_filter,\
                                self.filter_length, self.percent_vocab_size,\
                                self.nb_epoch)] = val 

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

    
        tr_X, tr_Y, ts_X, ts_Y, val_X, val_Y = getData(self.percent_vocab_size)
        self.model = self.runCNNBaseline (1, latent_space_size)
        self.model.fit(tr_X, tr_Y, batch_size=self.batch_size,\
                                 nb_epoch=self.nb_epoch, validation_data= (val_X, val_Y))
        result = pd.DataFrame()
        result['predicted_values'] = logit.predict(test_X)
        result['true_values'] = test_Y

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