#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os 
import re
import pickle
import gensim 
import logging 
import numpy as np 
from skipThought.training import vocab, train, tools 
from baselineRunner.BaselineRunner import BaselineRunner
from log_manager.log_config import Logger 
from utility.Utility import Utility
from summaryGenerator.SummaryGenerator import SummaryGenerator

class SkipThoughtPreLoadedRunner(BaselineRunner):
    def __init__(self, *args, **kwargs):
        """
        """
        BaselineRunner.__init__(self, *args, **kwargs)
        self.latReprName = "pretrained-skip-thought"
        self.rootdir = os.environ['SEN2VEC_DIR']
        self.postgresConnection.connectDatabase()
        self.utFunction = Utility("Text Utility")
        self.sentIDList = list()
        self.sentenceList = list()
        self.dataDir = os.environ['TRTESTFOLDER']
        self.system_id = 89
        self.sentReprFile = os.path.join(self.dataDir, "%s_sents_repr"%self.latReprName)

    def prepareData(self, pd):
        """
        """
        pass 

        

    def convert_to_str(self, vec):
        str_ = ""
        for val in vec: 
            str_ ="%s %0.3f"%(str_,val)
        return str_

    def runTheBaseline(self, rbase, latent_space_size):

        if rbase <=0: return 0 

        from skipThought import skipthoughts 
        model = skipthoughts.load_model()

        nSent = 0
        for result in self.postgresConnection.memoryEfficientSelect(["count(*)"],\
            ['sentence'], [], [], []):
            nSent = int (result[0][0])
        sent2vecFileRaw = open("%s_raw"%(self.sentReprFile),"w") 
        sent2vecFileRaw.write("%s %s%s"%(str(nSent), str(latent_space_size*2), os.linesep))

        sent2vecFile = open("%s.p"%(self.sentReprFile),"wb")
        sent2vec_dict = {}

        sent2vecFile_raw = open("%s_raw.p"%(self.sentReprFile),"wb")
        sent2vec_raw_dict = {}

        sentence_list = []
        for result in self.postgresConnection.memoryEfficientSelect(["id", "content"],\
         ["sentence"], [], [], ["id"]):
            for row_id in range(0,len(result)):
                id_ = result[row_id][0] 
                sentence = result[row_id][1]
                sentence_list.append(sentence)

        Logger.logr.info("Total Number of sentences = %i"%len(sentence_list))

        feature_map = skipthoughts.encode (model, sentence_list)

        start_id = 0
        for result in self.postgresConnection.memoryEfficientSelect(["id", "content"],\
         ["sentence"], [], [], ["id"]):
            for row_id in range(0,len(result)):
                id_ = result[row_id][0] 
                vec = feature_map[start_id]
                start_id = start_id + 1
                sent2vecFileRaw.write("%s "%(str(id_))) 
                vec_str = self.convert_to_str(vec)
           
                sent2vec_raw_dict[id_] = vec 

                sent2vecFileRaw.write("%s%s"%(vec_str, os.linesep))
                sent2vec_dict[id_] = vec /  ( np.linalg.norm(vec) +  1e-6)

        Logger.logr.info("Total Number of Sentences written=%i", len(sent2vec_dict))            
        pickle.dump(sent2vec_dict, sent2vecFile)    
        pickle.dump(sent2vec_raw_dict, sent2vecFile_raw)    

        sent2vecFile_raw.close()    
        sent2vecFile.close()

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
