#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os 
import sys 
import networkx as nx 
import pickle
import math 
import operator 
import multiprocessing 
import subprocess 
import numpy as np 
import gensim
from utility.Utility import Utility
from gensim.models import Word2Vec
from log_manager.log_config import Logger 
from gensim.models import Doc2Vec
from baselineRunner.BaselineRunner import BaselineRunner
from word2vec.WordDoc2Vec import WordDoc2Vec
from summaryGenerator.SummaryGenerator import SummaryGenerator

label_sent = lambda id_: 'SENT_%s' %(id_)


class FastSentVariantRunner(BaselineRunner): 

    def __init__(self, *args, **kwargs):
        BaselineRunner.__init__(self, *args, **kwargs)

        self.dataDir = os.environ['TRTESTFOLDER']
        self.latReprName = "con_s2v_s"
        self.full_data = int (os.environ["FULL_DATA"])
        self.lambda_val = float(os.environ['LAMBDA'])
        self.dbow_only = int(1)
        self.window = str(10)
        self.postgresConnection.connectDatabase()
        self.sentenceList = []
        self.utFunction = Utility("Text Utility")

        self.system_id = 85

        if self.dbow_only == 0:
            self.latReprName = "con_s2v_s"
        else:
            self.latReprName = "con_s2v_s_dbow_only"

        if self.lambda_val > 0.0:
            self.latReprName = "%s_%s"%(self.latReprName,"regularized")
        else:
            self.latReprName = "%s_%s"%(self.latReprName,"general")

        if self.full_data == 0:
            self.latReprName = "%s_%s"%(self.latReprName,"rnd")
        else:
            self.latReprName = "%s_%s"%(self.latReprName,"full")


        self.fastsentReprFile = os.path.join(self.dataDir, "%s_sents_repr"%self.latReprName)
        self.sentsFile = os.path.join(self.dataDir, "%s_sents"%self.latReprName)
       
      
        
    def insertNeighbors(self, sentenceList, nbr_file):
        for pos in range(0, len(sentenceList)):
            nbr_file.write("%s "%label_sent(sentenceList[pos]))
            if pos -1 >= 0:
                nbr_file.write("%s "%label_sent(sentenceList[pos-1]))
            else:
                nbr_file.write("%s "%("-1"))

            if pos+1 < len(sentenceList):
                nbr_file.write("%s "%label_sent(sentenceList[pos+1]))
            else:
                nbr_file.write("%s "%("-1"))
            nbr_file.write(os.linesep)
    

    def prepareSentsFile(self):
        sentfiletoWrite = open("%s.txt"%(self.sentsFile),"w")
        for result in self.postgresConnection.memoryEfficientSelect(["id","content"],\
             ["sentence"], [], [], ["id"]):
            for row_id in range(0,len(result)):
                id_ = result[row_id][0]
                content = gensim.utils.to_unicode(result[row_id][1].strip())
                content = self.utFunction.normalizeText(content, remove_stopwords=0)
                sentfiletoWrite.write("%s %s%s"%(label_sent(id_),' '.join(content), os.linesep))
            sentfiletoWrite.flush()
        sentfiletoWrite.close()

    def prepareData(self, pd):
        """
        This function will generate the context based on the 
        google's fast sent idea or idea from Felix Hill's 
        sentence representation paper. 
        """    
        if pd <=0 : return 0 

        self.prepareSentsFile() 
        nbr_file = open(os.path.join(self.dataDir, "%s%s"%(self.latReprName,"_nbr")), "w")

        nSent = 0
        for result in self.postgresConnection.memoryEfficientSelect(["count(*)"],\
            ['sentence'], [], [], []):
            nSent = int (result[0][0])

        nbr_file.write("%s 1 2%s"%(str(nSent),os.linesep))

        for doc_result in self.postgresConnection.memoryEfficientSelect(["id"],\
            ["document"], [], [], ["id"]):
            for row_id in range(0,len(doc_result)):
                document_id = doc_result[row_id][0]
                #Logger.logr.info("Working for Document id =%i", doc_result[row_id][0])
                self.sentenceList = []
                for sentence_result in self.postgresConnection.memoryEfficientSelect(\
                    ['id'],['sentence'],[["doc_id","=",document_id]],[],['id']):
                    for inrow_id in range(0, len(sentence_result)):
                        sentence_id = int(sentence_result[inrow_id][0])
                        self.sentenceList.append(sentence_id)
                self.insertNeighbors(self.sentenceList, nbr_file)


    def convert_to_str(self, vec):
        str_ = ""
        for val in vec: 
            str_ ="%s %0.3f"%(str_,val)
        return str_


    def runTheBaseline(self, rbase, latent_space_size):
        """
        Should we also optimize for window size?
        """
        if rbase <= 0: return 0 


        wordDoc2Vec = WordDoc2Vec()
        wPDict = wordDoc2Vec.buildWordDoc2VecParamDict()
        wPDict["cbow"] = str(0) 
        wPDict["output"] = os.path.join(self.dataDir , "%s_raw_DBOW"%self.latReprName)
        wPDict["sentence-vectors"] = str(1)
        wPDict["min-count"] = str(0)
        wPDict["window"] = str(self.window)
        wPDict["train"] = "%s.txt"%self.sentsFile
    
        wPDict["full_data"] = str(self.full_data)
        wPDict["lambda"] = str(self.lambda_val)


        if self.dbow_only ==1:
            wPDict["size"]= str(latent_space_size *2 )
        else:
            wPDict["size"]= str(latent_space_size)

        args = []
        neighborFile = os.path.join(self.dataDir,"%s_nbr"%(self.latReprName))
        wPDict["neighborFile"] = neighborFile
 
        args = wordDoc2Vec.buildArgListforW2VWithNeighbors(wPDict, 3)
        self._runProcess(args)
        jointvecModel = Doc2Vec.load_word2vec_format(wPDict["output"], binary=False)


        jointvecModelDM = ""
        if self.dbow_only ==0:
            wPDict["cbow"] = str(1) 
            wPDict["output"] = os.path.join(self.dataDir,"%s_raw_DM"%self.latReprName)
            args = wordDoc2Vec.buildArgListforW2VWithNeighbors(wPDict, 3)
            self._runProcess(args)
            jointvecModelDM = Doc2Vec.load_word2vec_format(wPDict["output"], binary=False)  

        jointvecFile = open("%s.p"%(self.fastsentReprFile),"wb")
        jointvec_dict = {}

        jointvecFile_raw = open("%s_raw.p"%(self.fastsentReprFile),"wb")
        jointvec_raw_dict = {}

        for result in self.postgresConnection.memoryEfficientSelect(["id"],\
             ["sentence"], [], [], ["id"]):
            for row_id in range(0,len(result)):
                id_ = result[row_id][0] 
                vec1 = jointvecModel[label_sent(id_)]
                if self.dbow_only ==0:
                    vec2 = jointvecModelDM[label_sent(id_)]
                    vec = np.hstack((vec1,vec2))
                    jointvec_raw_dict[id_] = vec        
                else:
                    vec = vec1
                    jointvec_raw_dict[id_] = vec

                jointvec_dict[id_] = vec /  ( np.linalg.norm(vec) +  1e-6)
                
        Logger.logr.info("Total Number of Sentences written=%i", len(jointvec_raw_dict))            
        pickle.dump(jointvec_dict, jointvecFile)    
        pickle.dump(jointvec_raw_dict, jointvecFile_raw)    

        jointvecFile_raw.close()    
        jointvecFile.close()

    def generateSummary(self, gs, methodId, filePrefix,\
         lambda_val=1.0, diversity=False):
        if gs <= 0: return 0
        fastsentvecFile = open("%s.p"%(self.fastsentReprFile),"rb")
        fastsentvDict = pickle.load (fastsentvecFile)

        summGen = SummaryGenerator (diverse_summ=diversity,\
             postgres_connection = self.postgresConnection,\
             lambda_val = lambda_val)
        summGen.populateSummary(methodId, fastsentvDict)

        # Need a method id for the joint 
    
    def runEvaluationTask(self):
        summaryMethodID = 2 
        what_for =""

        try: 
            what_for = os.environ['VALID_FOR'].lower()
        except:
            what_for = os.environ['TEST_FOR'].lower()

        vDict  = {}
        if  "rank" in what_for:
            vecFile = open("%s.p"%(self.fastsentReprFile),"rb")
            vDict = pickle.load(vecFile)
        else:
            vecFile_raw = open("%s_raw.p"%(self.fastsentReprFile),"rb")
            vDict = pickle.load(vecFile_raw)

        Logger.logr.info ("Performing evaluation for %s"%what_for)
        self.performEvaluation(summaryMethodID, self.latReprName, vDict)
        
    def doHouseKeeping(self):
        """
        Here, we destroy the database connection.
        """
        self.postgresConnection.disconnectDatabase()