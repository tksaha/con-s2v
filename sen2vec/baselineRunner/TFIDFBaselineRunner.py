#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os 
import sys 
import pickle 
import gensim
import scipy.stats
from utility.Utility import Utility
from baselineRunner.BaselineRunner import BaselineRunner



class TFIDFBaselineRunner(BaselineRunner):
    def __init__(self, *args, **kwargs):
        """
        """
        BaselineRunner.__init__(self, *args, **kwargs)
        self.system_id  = 1
        self.latReprName = "TFIDF"
        self.rootdir = os.environ['SEN2VEC_DIR']
        self.postgresConnection.connectDatabase()
        self.utFunction = Utility("Text Utility")

    def prepareData(self, pd):
        pass 

    def runTheBaseline(self, rbase):
        pass 


    # This function will go inside evaluation
    def evaluateRankCorrelation(self, dataset):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise  import cosine_similarity


        test_pair_file = open(os.path.join(self.rootdir,"Data/test_pair_%s.p"%(dataset)), "rb")
        test_dict = pickle.load(test_pair_file)

        original_val = []
        computed_val = []
        for k, val in test_dict.items():
            corpus = []
            for sentence_result in self.postgresConnection.memoryEfficientSelect(\
                    ['id','content'],['sentence'],[["id","=",k[0]]],[],[]):
                content = sentence_result[0][1]
                content = gensim.utils.to_unicode(content)
                content = self.utFunction.normalizeText(content, remove_stopwords=0)
                corpus.append(' '.join(content))

            for sentence_result in self.postgresConnection.memoryEfficientSelect(\
                    ['id','content'],['sentence'],[["id","=",k[1]]],[],[]):
                content = sentence_result[0][1]
                content = gensim.utils.to_unicode(content)
                content = self.utFunction.normalizeText(content, remove_stopwords=0)
                corpus.append(' '.join(content))

            vectorizer = TfidfVectorizer(stop_words='english')
            vecs  = vectorizer.fit_transform(corpus)
            original_val.append(val)
            computed_val.append(cosine_similarity(vecs[0:1],vecs)[0][1])

        if os.environ['TEST_AND_TRAIN'] =="YES":
            train_pair_file = open(os.path.join(self.rootdir,"Data/train_pair_%s.p"%(dataset)), "rb")
            train_dict = pickle.load(train_pair_file)
            for k, val in train_dict.items():
                original_val.append(val)
                corpus = []
                for sentence_result in self.postgresConnection.memoryEfficientSelect(\
                    ['id','content'],['sentence'],[["id","=",k[0]]],[],[]):
                    content = sentence_result[0][1]
                    content = gensim.utils.to_unicode(content)
                    content = self.utFunction.normalizeText(content, remove_stopwords=0)
                    corpus.append(' '.join(content))

                for sentence_result in self.postgresConnection.memoryEfficientSelect(\
                    ['id','content'],['sentence'],[["id","=",k[1]]],[],[]):
                    content = sentence_result[0][1]
                    content = gensim.utils.to_unicode(content)
                    content = self.utFunction.normalizeText(content, remove_stopwords=0)
                    corpus.append(' '.join(content))
                    

                vectorizer = TfidfVectorizer(stop_words='english')
                vecs  = vectorizer.fit_transform(corpus)

                computed_val.append(cosine_similarity(vecs[0:1], vecs)[0][1])

        Logger.logr.info (len(original_val))
        Logger.logr.info (len(computed_val))

        sp = scipy.stats.spearmanr(original_val,computed_val)[0]
        pearson = scipy.stats.pearsonr(original_val,computed_val)[0]
        return sp, pearson

    def runEvaluationTask(self):
       
        vDict = {}
        summaryMethodID = 2
        
        if os.environ['EVAL']=='TEST' and os.environ['TEST_FOR']=='CLASS':
            self._runClassificationTF(summaryMethodID, "TFIDF", vDict)
        else:
            self._runClusteringTF(summaryMethodID, "TFIDF", vDict)




