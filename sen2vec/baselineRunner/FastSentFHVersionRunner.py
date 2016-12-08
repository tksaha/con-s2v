#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os 
import logging 
import re
import numpy as np 
import gensim 
from baselineRunner.BaselineRunner import BaselineRunner
from fastsent.fastsent  import FastSent 
from log_manager.log_config import Logger 
from utility.Utility import Utility

class FastSentFHVersionRunner(BaselineRunner):
	
	def __init__(self, *args, **kwargs):
        BaselineRunner.__init__(self, *args, **kwargs)
        self.sentIDList = list()
        self.sentenceList = list()
        self.cores = multiprocessing.cpu_count()
        self.dataDir = os.environ['TRTESTFOLDER']
        self.latReprName = 'fastsent'


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
 		# What is auto-encode ?
 		model = FastSent(workers=self.cores*2, fastsent_mean=0, min_count=0,\
 			 size= latent_space_size , autoencode=False, sample=1e-4) 
 		model.build_vocab(self.sentenceList)
		model.train(self.sentenceList, chunksize=1000)
		model.save(os.path.join(self.dataDir,"%s_repr"%self.latReprName))
    	

 	def generateSummary():
 		pass

 	def runEvaluationTask():
 		pass

 	def doHouseKeeping():
 		pass 
