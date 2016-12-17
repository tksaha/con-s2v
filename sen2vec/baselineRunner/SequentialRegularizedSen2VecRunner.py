#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os 
import sys 
import pickle
import math 
import operator 
import multiprocessing 
import subprocess 
import numpy as np 
import scipy.stats
import networkx as nx 
import gensim 
from utility.Utility import Utility
from gensim.models import Word2Vec
from utility.Utility import Utility
from log_manager.log_config import Logger 
from baselineRunner.BaselineRunner import BaselineRunner
from word2vec.WordDoc2Vec import WordDoc2Vec
from summaryGenerator.SummaryGenerator import SummaryGenerator


label_sent = lambda id_: 'SENT_%s' %(id_)


class SequentialRegularizedSen2VecRunner(BaselineRunner): 

    def __init__(self, *args, **kwargs):
        BaselineRunner.__init__(self, *args, **kwargs)
        self.dataDir = os.environ['TRTESTFOLDER']
        self.window_size = str(10)
        self.latReprName = "seq_reg_s2v"
        self.sentsFile = os.path.join(self.dataDir, "%s_sents"%self.latReprName)
        self.seqregunw_beta = str(1.0)
        self.postgresConnection.connectDatabase()
        self.rootdir = os.environ['SEN2VEC_DIR']
        self.sentenceList = list()
        self.seqregsen2vReprFile = os.path.join(self.dataDir, "%s_repr"%self.latReprName)
        self.system_id = 83
        self.utFunction = Utility("Text Utility")
    
    def __insertNeighbors(self, sentenceList, nbr_file):
        for pos in range(0, len(sentenceList)):
            nbr_file.write("%s "%label_sent(sentenceList[pos]))
            if pos -1 >= 0:
                nbr_file.write("%s %s "%(label_sent(sentenceList[pos-1]), "1.0"))
            else:
                nbr_file.write("%s %s "%("-1", "1.0"))

            if pos+1 < len(sentenceList):
                nbr_file.write("%s %s "%(label_sent(sentenceList[pos+1]), "1.0"))
            else:
                nbr_file.write("%s %s "%("-1", "1.0"))
            nbr_file.write(os.linesep)

    def __write_neighbors (self, max_neighbor, file_to_write):

        nSent = 0
        for result in self.postgresConnection.memoryEfficientSelect(["count(*)"],\
            ['sentence'], [], [], []):
            nSent = int (result[0][0])
        file_to_write.write("%s %s%s"%(nSent,max_neighbor, os.linesep))
        

        for doc_result in self.postgresConnection.memoryEfficientSelect(["id"],\
            ["document"], [], [], ["id"]):
            for row_id in range(0,len(doc_result)):
                document_id = doc_result[row_id][0]
                self.sentenceList = []
                for sentence_result in self.postgresConnection.memoryEfficientSelect(\
                    ['id'],['sentence'],[["doc_id","=",document_id]],[],['id']):
                    for inrow_id in range(0, len(sentence_result)):
                        sentence_id = int(sentence_result[inrow_id][0])
                        self.sentenceList.append(sentence_id)
                self.__insertNeighbors(self.sentenceList, file_to_write)

        file_to_write.flush()
        file_to_write.close()

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
    
        if pd <= 0: return 0 
        self.prepareSentsFile()
        max_neighbor = 2
        neighbor_file_unw = open("%s_neighbor_unw.txt"%\
                self.seqregsen2vReprFile, "w")
        self.__write_neighbors (max_neighbor, neighbor_file_unw)
       
    def __dumpVecs(self, reprFile, vecFile, vecRawFile):

        vModel = Word2Vec.load_word2vec_format(reprFile, binary=False)
        vec_dict = {}
        vec_dict_raw = {}

        for sentence_result in self.postgresConnection.memoryEfficientSelect(\
                ['id'],['sentence'],[],[],['id']):
            for inrow_id in range(0, len(sentence_result)):
                nodes = int(sentence_result[inrow_id][0])
                vec = vModel[label_sent(str(nodes))]
                vec_dict_raw[int(nodes)] = vec 
                vec_dict[int(nodes)] = vec /  ( np.linalg.norm(vec) +  1e-6)

        pickle.dump(vec_dict, vecFile)
        pickle.dump(vec_dict_raw, vecRawFile)

    def runTheBaseline(self, rbase, latent_space_size):
        if rbase <= 0: return 0 
        wordDoc2Vec = WordDoc2Vec()
        wPDict = wordDoc2Vec.buildWordDoc2VecParamDict()
        wPDict["cbow"] = str(0) 
        wPDict["sentence-vectors"] = str(1)
        wPDict["min-count"] = str(0)
        wPDict["train"] = "%s.txt"%self.sentsFile
        wPDict["window"] = self.window_size
        
        wPDict["size"]= str(latent_space_size * 2)
        args = []

        neighborFile =  "%s_neighbor_unw.txt"%(self.seqregsen2vReprFile)
        wPDict["output"] = "%s_neighbor_unw"%(self.seqregsen2vReprFile)
        wPDict["neighborFile"], wPDict["beta"] = neighborFile, str(self.seqregunw_beta)

        args = wordDoc2Vec.buildArgListforW2VWithNeighbors(wPDict, 2)
        self._runProcess (args)
        self.__dumpVecs(wPDict["output"],\
                open("%s.p"%wPDict["output"], "wb"),\
                open("%s_raw.p"%wPDict["output"], "wb"))
    

    def generateSummary(self, gs, methodId, filePrefix,\
         lambda_val=1.0, diversity=False):
        if gs <= 0: return 0
        vecFile = open("%s%s.p"%(self.seqregsen2vReprFile,\
             filePrefix),"rb")
        vDict = pickle.load (vecFile)
        
        summGen = SummaryGenerator (diverse_summ=diversity,\
             postgres_connection = self.postgresConnection,\
             lambda_val = lambda_val)

        summGen.populateSummary(methodId, vDict)
        

    def __runEval(self, summaryMethodID, vecFileName, reprName):
       
        what_for =""
        try: 
            what_for = os.environ['VALID_FOR'].lower()
        except:
            what_for = os.environ['TEST_FOR'].lower()

        vDict  = {}
        if  "rank" in what_for:
            vecFile = open("%s.p"%(vecFileName),"rb")
            vDict = pickle.load(vecFile)
        else:
            vecFile_raw = open("%s_raw.p"%(vecFileName),"rb")
            vDict = pickle.load(vecFile_raw)

        Logger.logr.info ("Performing evaluation for %s"%what_for)
        Logger.logr.info("Regularized Dictionary has %i objects"%len(vDict))

        self.performEvaluation(summaryMethodID, reprName, vDict)



    def runEvaluationTask(self):
        summaryMethodID = 2 
        Logger.logr.info("Starting Sequential Regularized Sentence 2 Vector Evaluation")
        
        regvecFile = "%s_neighbor_unw"%(self.seqregsen2vReprFile)
        reprName = "%s_neighbor_unw"%self.latReprName
        self.__runEval(summaryMethodID, regvecFile, reprName)

        
    def doHouseKeeping(self):
        """
        Here, we destroy the database connection.
        """
        self.postgresConnection.disconnectDatabase()