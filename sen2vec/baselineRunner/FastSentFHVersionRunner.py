#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os 
import logging 
import re
import numpy as np 
import gensim 
import multiprocessing
from baselineRunner.BaselineRunner import BaselineRunner
from fastsent.fastsent  import FastSent 
from log_manager.log_config import Logger 
from utility.Utility import Utility


class MySentences(object):
    def __init__(self, sentenceList):
        self.sentenceList = sentenceList

    def __iter__(self):
        for line in self.sentenceList:
            yield line.split()


class FastSentFHVersionRunner(BaselineRunner):
    
    def __init__(self, *args, **kwargs):
        BaselineRunner.__init__(self, *args, **kwargs)
        self.sentIDList = list()
        self.sentenceList = list()
        self.cores = multiprocessing.cpu_count()
        self.window = str(10)
        self.dataDir = os.environ['TRTESTFOLDER']
        self.latReprName = 'felixhillfastsent'
        self.utFunction = Utility("Text Utility")
        self.postgresConnection.connectDatabase()

    def prepareData(self, pd):
        Logger.logr.info ("Preparing Data for FastSent")
        for doc_result in self.postgresConnection.memoryEfficientSelect(["id"],\
            ["document"], [], [], ["id"]):
            for row_id in range(0,len(doc_result)):
                document_id = doc_result[row_id][0]
                for sentence_result in self.postgresConnection.memoryEfficientSelect(\
                    ['id','content'],['sentence'],[["doc_id","=",document_id]],[],['id']):
                    for inrow_id in range(0, len(sentence_result)):
                        sentence_id = int(sentence_result[inrow_id][0])
                        sentence = sentence_result[inrow_id][1]
                        content = gensim.utils.to_unicode(sentence) 
                        content = self.utFunction.normalizeText(content, remove_stopwords=0)
                        self.sentenceList.append(' '.join(content))
                        self.sentIDList.append(sentence_id) 

        
    def runTheBaseline(self, rbase, latent_space_size):
        # What is auto-encode +AE version, AE =0
        sentences = MySentences(self.sentenceList)

        model = FastSent(sentences, size=latent_space_size,\
            window=self.window, min_count=0, workers=self.cores*2, sample=1e-4) 
        model.build_vocab(sentences)
        model.train(sentences, chunksize=1000)
        model.save_fastsent_format(os.path.join(self.dataDir,\
            "%s_repr"%self.latReprName), binary=False)  


              

    def generateSummary():
        pass

    def runEvaluationTask():
        

    def doHouseKeeping():
        pass 
