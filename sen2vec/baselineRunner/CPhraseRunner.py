#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os 
import re
import pickle 
import gensim 
import logging 
import numpy as np
import multiprocessing
from gensim.models import Doc2Vec
from utility.Utility import Utility
from log_manager.log_config import Logger 
from baselineRunner.BaselineRunner import BaselineRunner
from summaryGenerator.SummaryGenerator import SummaryGenerator



# Please put the precomputed cphrase embeddings 
# under the Data Directory

class CPhraseRunner(BaselineRunner):
    
    def __init__(self, dbstring, **kwargs):
        BaselineRunner.__init__(self, dbstring, **kwargs)
        self.system_id = 90
        self.sentIDList = list()
        self.sentenceList = list()
        self.latReprName = 'cphrase'
        self.cores = multiprocessing.cpu_count()
        self.dataDir = os.environ['TRTESTFOLDER']
        self.utFunction = Utility("Text Utility")
        self.postgresConnection.connectDatabase()
        self.cphraseFileDir = os.path.join(self.dataDir, "cphrase_data")
        self.cphraseEmb = os.path.join(self.cphraseFileDir, "cphrase.txt")
        self.outembFile = os.path.join(self.dataDir, "%s_repr"%(self.latReprName))

   
    def prepareData(self, pd):
        pass 

    def runTheBaseline(self, rbase, latent_space_size):
        """
        """
        Logger.logr.info("Started Loading CPhrase Pretrained Vectors")
        cphraseModel = Doc2Vec.load_word2vec_format(self.cphraseEmb, binary=False)

        cphrasevecFile = open("%s.p"%(self.outembFile),"wb")
        cphrasevec_dict = {}

        cphrasevecFile_raw = open("%s_raw.p"%(self.outembFile),"wb")
        cphrasevec_raw_dict = {}

        for result in self.postgresConnection.memoryEfficientSelect(["id", "content"],\
             ["sentence"], [], [], ["id"]):
            for row_id in range(0,len(result)):
                id_ = result[row_id][0] 
                sentence = result[row_id][1]

                content = gensim.utils.to_unicode(sentence) 
                content = self.utFunction.normalizeTextNoStemming(content,\
                    remove_stopwords=0)

                if len(content) == 0: continue 
                vec = np.zeros(latent_space_size)

                for word in content: 
                    try: 
                        vec += cphraseModel[word]
                    except:
                        pass 

                cphrasevec_raw_dict[id_] = vec 
                cphrasevec_dict[id_] = vec /  ( np.linalg.norm(vec) +  1e-6)

        Logger.logr.info("Total Number of Sentences written=%i", len(cphrasevec_dict))
        pickle.dump(cphrasevec_dict, cphrasevecFile)    
        pickle.dump(cphrasevec_raw_dict, cphrasevecFile_raw)    

        cphrasevecFile_raw.close()    
        cphrasevecFile.close()



    def generateSummary(self, gs, methodId, filePrefix,\
         lambda_val=1.0, diversity=False):

        if gs <= 0: return 0
        outFile = os.path.join(self.dataDir,\
            "%s_repr"%self.latReprName)
        vecFile = open("%s.p"%(outFile),"rb")
        vDict = pickle.load (vecFile)

        summGen = SummaryGenerator (diverse_summ=diversity,\
             postgres_connection = self.postgresConnection,\
             lambda_val = lambda_val)

        summGen.populateSummary(methodId, vDict)

    def runEvaluationTask(self):
        summaryMethodID = 2
        what_for =""
        try: 
            what_for = os.environ['VALID_FOR'].lower()
        except:
            what_for = os.environ['TEST_FOR'].lower()

        vDict = {}

        if  "rank" in what_for:
            vecFile = open("%s.p"%(os.path.join(self.dataDir,"%s_repr"%self.latReprName)),"rb")
            vDict = pickle.load(vecFile)
        else:
            vecFile_raw = open("%s.p"%(os.path.join(self.dataDir,"%s_repr"%self.latReprName)),"rb")
            vDict = pickle.load(vecFile_raw)

        Logger.logr.info ("Performing evaluation for %s"%what_for)
        self.performEvaluation(summaryMethodID, self.latReprName, vDict)


    def doHouseKeeping(self):
        self.postgresConnection.disconnectDatabase()