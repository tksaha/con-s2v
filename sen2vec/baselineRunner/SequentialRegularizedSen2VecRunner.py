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
from gensim.models import Word2Vec
from log_manager.log_config import Logger 
from baselineRunner.BaselineRunner import BaselineRunner
from word2vec.WordDoc2Vec import WordDoc2Vec
from summaryGenerator.SummaryGenerator import SummaryGenerator


label_sent = lambda id_: 'SENT_%s' %(id_)


class SequentialRegularizedSen2VecRunner(BaselineRunner): 

    def __init__(self, *args, **kwargs):
        BaselineRunner.__init__(self, *args, **kwargs)
        self.dataDir = os.environ['TRTESTFOLDER']
        self.sentsFile = os.environ['P2VCEXECSENTFILE']
        self.window_size = str(10)
        self.latReprName = "seq_reg_s2v"
        self.seqregunw = str(1.0)
        self.postgresConnection.connectDatabase()
        self.rootdir = os.environ['SEN2VEC_DIR']
        self.seqregsen2vReprFile = os.path.join(self.dataDir, self.latReprName)
    
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

    def prepareData(self, pd):
    
        if pd <= 0: return 0 
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
        wPDict["neighborFile"], wPDict["beta"] = neighborFile, str(self.regBetaUNW)

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
        vecFile = open("%s_raw.p"%vecFileName, "rb")
        vDict = pickle.load (vecFile)
        Logger.logr.info("Regularized Dictionary has %i objects"%len(vDict))

        if os.environ['EVAL']=='VALID' and os.environ['VALID_FOR']=='CLASS':
            self._runClassificationValidation(summaryMethodID, "%s_raw"%reprName, vDict)
        elif os.environ['EVAL']=='VALID' and os.environ['VALID_FOR']=='CLUST':
            self._runClusteringValidation(summaryMethodID, "%s_raw"%reprName, vDict)
        elif os.environ['EVAL']=='TEST' and os.environ['TEST_FOR']=='CLASS':
            self._runClassification(summaryMethodID, "%s_raw"%reprName, vDict)
        else:
            self._runClustering(summaryMethodID, "%s_raw"%reprName, vDict)


    def evaluateRankCorrelation(self,dataset):
        vecFile =  open("%s%s.p"%(self.seqregsen2vReprFile,\
             "_neighbor_unw"),"rb")
        vDict = pickle.load (vecFile)


        if os.environ['EVAL']=='VALID':
            validation_pair_file = open(os.path.join(self.rootdir,"Data/validation_pair_%s.p"%(dataset)), "rb")
            val_dict = pickle.load(validation_pair_file)

            original_val = []
            computed_val = []
            for k, val in val_dict.items():
                original_val.append(val)
                computed_val.append(np.inner(vDict[(k[0])],vDict[(k[1])]))
            return scipy.stats.spearmanr(original_val,computed_val)[0]
        else:
            test_pair_file = open(os.path.join(self.rootdir,"Data/test_pair_%s.p"%(dataset)), "rb")
            test_dict = pickle.load(test_pair_file)

            original_val = []
            computed_val = []
            for k, val in test_dict.items():
                original_val.append(val)
                computed_val.append(np.inner(vDict[(k[0])],vDict[(k[1])]))

            if os.environ['TEST_AND_TRAIN'] =="YES":
                train_pair_file = open(os.path.join(self.rootdir,"Data/train_pair_%s.p"%(dataset)), "rb")
                train_dict = pickle.load(train_pair_file)
                for k, val in train_dict.items():
                    original_val.append(val)
                    computed_val.append(np.inner(vDict[(k[0])],vDict[(k[1])]))

            Logger.logr.info (len(original_val))
            Logger.logr.info (len(computed_val))
            sp = scipy.stats.spearmanr(original_val,computed_val)[0]
            pearson = scipy.stats.pearsonr(original_val,computed_val)[0]
            return sp, pearson

            
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