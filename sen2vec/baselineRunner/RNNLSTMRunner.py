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



class RNNLSTMRunner (BaselineRunner):
	 def __init__(self, *args, **kwargs):
        """
        """
        BaselineRunner.__init__(self, *args, **kwargs)
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
        self.word_counter = Counter() 


        np.random.seed(2016)

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

    def encodeLabel (self, Y):
        encoder = LabelEncoder()
        encoder.fit(Y)
        encoded_Y = encoder.transform(Y)
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
        train_x = []
        for sentence in train_sentences:
            train_x.append(self.getlistOfIndexes(sentence))

        test_x = []
        for sentence in test_sentences:
            test_x.append(self.getlistOfIndexes(sentence))

        val_x = []
        for sentence in valid_sentences: 
            val_x.append(self.getlistOfIndexes(sentence))

        train_x = sequence.pad_sequences(train_x, maxlen=self.maxlen)
        test_x = sequence.pad_sequences(test_x, maxlen=self.maxlen)
        val_x = sequence.pad_sequences(val_x, maxlen=self.maxlen)

        self.n_classes = len(np.unique(train_labels))

        train_labels = np_utils.to_categorical(self.encodeLabel(train_labels), self.n_classes)
        test_labels  = np_utils.to_categorical(self.encodeLabel(test_labels), self.n_classes)
        valid_labels = np_utils.to_categorical(self.encodeLabel(valid_labels), self.n_classes)
        Logger.logr.info ("Total Number of Classes =%i" %self.n_classes)

        return train_x, train_labels, test_x, test_labels, val_x, valid_labels, total_to_take

    def runEvaluationTask(self,  rbase, latent_space_size):
        # Run the LSTM baseline 
      
        metric = {}

        for self.batch_size in [16, 32, 64, 128]:
            for self.percent_vocab_size in [80, 85, 90, 95]:
                tr_X, tr_Y, ts_X, ts_Y, val_X, val_Y, val_Y_prime,\
                     max_feat = self.getData(self.percent_vocab_size)
                for self.nb_epoch in [2, 5, 7]:
                    self.max_features = max_feat
                    self.runLSTMBaseline (1, latent_space_size)
                    self.model.fit(tr_X, tr_Y, batch_size=self.batch_size,\
                         nb_epoch=self.nb_epoch, shuffle=True,\
                         validation_data= (val_X, val_Y_prime))
                    result = pd.DataFrame()
                    result['predicted_values'] = self.model.predict_classes(val_X, batch_size=64)
                    result['true_values'] = val_Y

                    val = mt.f1_score(result['true_values'],\
                            result['predicted_values'], average = 'macro') 

                    metric[(self.batch_size, self.nb_filter,\
                        self.filter_length, self.percent_vocab_size,\
                        self.nb_epoch)] = val 
                    Logger.logr.info ("F1 value =%.4f"%val)

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

    
        tr_X, tr_Y, ts_X, ts_Y, val_X, val_Y, self.max_features = self.getData(self.percent_vocab_size)
        self.model = self.runCNNBaseline (1, latent_space_size)
        self.model.fit(tr_X, tr_Y, batch_size=self.batch_size,\
                                 nb_epoch=self.nb_epoch, validation_data= (val_X, val_Y))
        result = pd.DataFrame()
        result['predicted_values'] = self.model.predict_classes(test_X)
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