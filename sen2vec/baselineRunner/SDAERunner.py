#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os 
import sys
import pickle
import gensim 
import numpy as np
import pandas as pd 
import sklearn.metrics as mt
from collections import Counter 
from log_manager.log_config import Logger 
from utility.Utility import Utility
from SDAE.desent import train


class SDAERunner(BaselineRunner):
	def __init__(self, *args, **kwargs):
		"""
		"""
		BaselineRunner.__init__(self, *args, **kwargs)
		self.postgresConnection.connectDatabase()
		self.latReprName = "sdae"
		self.system_id = 88

		self.dim_word = 300, # word vector dimensionality
        self.dim = 600, # the number of RNN units
        self.patience = 10,
        self.max_epochs = 5000,
        self.dispFreq  = 100,
        self.corruption = ['_mask', '_shuffle'],
        self.corruption_prob = [0.1, 0.1],
        self.decay_c = 0., 
        self.lrate = 0.01, 
        self.clip_c = -1.,
        self.param_noise = 0.,
        self.n_words = 30000,
        self.maxlen = 100, # maximum length of the description
        self.optimizer = 'adam', 
        self.batch_size = 16,
        self.valid_batch_size = 16,
        self.saveto = 'model.npz',
        self.validFreq = 1000,
        self.saveFreq = 1000, # save the parameters after every saveFreq updates
        self.encoder = 'gru',
        self.decoder = 'gru_cond',
        self.dataset = 'wiki',
        self.use_preemb = False,
        self.embeddings = None,
        self.dictionary = os.path.join(self.dataDir,\
        		"%s_%s_dictionary.p"%(os.environ['DATASET'], self.latReprName)),
        self.valid_text = ,
        self.test_text =  ,
        self.use_dropout = False,
        self.reload_ = False

    # The following three function has been copied from
    # SDAE (Felix Hill). I have modified the functions 
    # according to my need.
	def add_line(self.textline, D):
	    content = gensim.utils.to_unicode(textline) 
	    words = self.utFunction.normalizeText(content, remove_stopwords=0)
	    for w in words:
	        D[w] += 1
	    return D
	    
	def save_dictionary(self, outpath, D):
	    with open(outpath,'wb') as out:
 	       pickle.dump(D,out)


    def prepareData(self, pd):
    	D = defaultdict(int)
    	Logger.logr.info ("Preparing Data for Skip-Thought")
        for doc_result in self.postgresConnection.memoryEfficientSelect(["id"],\
            ["document"], [], [], ["id"]):
            for row_id in range(0,len(doc_result)):
                document_id = doc_result[row_id][0]
                for sentence_result in self.postgresConnection.memoryEfficientSelect(\
                    ['id','content'],['sentence'],[["doc_id","=",document_id]],[],['id']):
                    for inrow_id in range(0, len(sentence_result)):
                        sentence_id = int(sentence_result[inrow_id][0])
                        sentence = sentence_result[inrow_id][1]
        				D = self.add_line(sentence, D)

        oFD = sorted_x = sorted(D.items(), key=operator.itemgetter(1),\
	         reverse=True)
	    for ii,(w,f) in enumerate(oFD):
	        D[w] = ii
	    
	    self.save_dictionary (self.dictionary, D)


	def runSDAE (self):
		trainerr, validerr, testerr = train(saveto=params['model'][0],
                                        reload_=params['reload'][0],
                                        corruption=params['corruption'][0],
                                        corruption_prob=params['corruption_prob'][0],
                                        dim_word=params['dim_word'][0],
                                        dim=params['dim'][0],
                                        n_words=params['n-words'][0],
                                        decay_c=params['decay-c'][0],
                                        param_noise=params['param-noise'][0],
                                        lrate=params['learning-rate'][0],
                                        optimizer=self.optimizer, 
                                        maxlen=100,
                                        batch_size=16,
                                        dictionary = params['dictionary'][0],
                                        valid_batch_size=16,
                                        validFreq=1000,
                                        dispFreq=1,
                                        saveFreq=1000,
                                        clip_c=params['clip-c'][0],
                                        encoder = self.encoder,
                                        decoder = self.decoder,
                                        use_preemb=params['use_preemb'][0],
                                        dataset='book', 
                                        use_dropout=params['use-dropout'][0],
                                        embeddings=params['embeddings'][0])

	def runTheBaseline(self, rbase, latent_space_size):
		pass 


	def generateSummary(self, gs, methodId, filePrefix,\
         lambda_val=1.0, diversity=False):

        if gs <= 0: return 0
        sent2vecFile = open("%s.p"%(self.sentReprFile),"rb")
        s2vDict = pickle.load (sent2vecFile)

        summGen = SummaryGenerator (diverse_summ=diversity,\
             postgres_connection = self.postgresConnection,\
             lambda_val = lambda_val)

        summGen.populateSummary(methodId, s2vDict)

    def runEvaluationTask(self):
        """
        Generate Summary sentences for each document. 
        Write sentence id and corresponding metadata 
        into a file. 
        """
        summaryMethodID = 2

        what_for =""
        try: 
            what_for = os.environ['VALID_FOR'].lower()
        except:
            what_for = os.environ['TEST_FOR'].lower()

        vDict  = {}
        if  "rank" in what_for:
            vecFile = open("%s.p"%(self.sentReprFile),"rb")
            vDict = pickle.load(vecFile)
        else:
            vecFile_raw = open("%s_raw.p"%(self.sentReprFile),"rb")
            vDict = pickle.load(vecFile_raw)

        Logger.logr.info("Total ids in dictionary =%i"%len(vDict))
        Logger.logr.info ("Performing evaluation for %s"%what_for)
        self.performEvaluation(summaryMethodID, self.latReprName, vDict)
       
        
    def doHouseKeeping(self):
        """
        Here, we destroy the database connection.
        """
        self.postgresConnection.disconnectDatabase()



     