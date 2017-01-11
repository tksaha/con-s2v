#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os 
import sys
import pickle
import gensim 
import operator
from collections import defaultdict
import numpy as np
import pandas as pd 
import sklearn.metrics as mt
from collections import Counter 
from log_manager.log_config import Logger 
from utility.Utility import Utility

from SDAE.desent import train
from SDAE.encoder import load_model, encode

from baselineRunner.BaselineRunner import BaselineRunner

class SAERunner(BaselineRunner):
    def __init__(self, *args, **kwargs):
        """
        """
        BaselineRunner.__init__(self, *args, **kwargs)
        self.postgresConnection.connectDatabase()
        self.latReprName = "sae"
        self.system_id = 91
        self.utFunction = Utility("Text Utility")
        self.dataDir = os.environ['TRTESTFOLDER']
        self.sentsFile = os.path.join(self.dataDir, "%s_sents"%self.latReprName)

        self.dim_word = 100 # word vector dimensionality
        self.dim = 2400 # the number of RNN units
        self.patience = 10
        self.max_epochs = 5
        self.dispFreq  = 100
        self.corruption = ['_mask', '_shuffle']
        self.corruption_prob = [0.0, 0.0]
        self.decay_c = 0. 
        self.lrate = 0.01 
        self.clip_c = -1.
        self.param_noise = 0.
        self.n_words = 30000
        self.maxlen = 100 # maximum length of the description
        self.optimizer = 'adam' 
        self.batch_size = 16
        self.valid_batch_size = 16
        self.saveto = os.path.join(self.dataDir,\
                 "model_%s_%s.npz"%(os.environ['DATASET'], self.latReprName))
        self.validFreq = 1000
        self.saveFreq = 1000 # save the parameters after every saveFreq updates
        self.encoder = 'gru'
        self.decoder = 'gru_cond'
        self.dataset = os.environ['DATASET']
        self.use_preemb = False
        self.embeddings = None
        self.dictionary = os.path.join(self.dataDir,\
                "%s_%s_dictionary.p"%(os.environ['DATASET'], self.latReprName))
        self.valid_text = None
        self.test_text =  None
        self.use_dropout = False
        self.reload_ = False
        self.outembFile = os.path.join(self.dataDir, "%s_repr"%(self.latReprName))

    # The following three function has been copied from
    # SDAE (Felix Hill). I have modified the functions 
    # according to my need.
    def add_line(self, textline, D):
        content = gensim.utils.to_unicode(textline) 
        words = self.utFunction.normalizeText(content, remove_stopwords=0)
        for w in words:
            D[w] += 1
        return D
        
    def save_dictionary(self, outpath, D):
        with open(outpath,'wb') as out:
           pickle.dump(D,out)

    def convert_to_str(self, vec):
        str_ = ""
        for val in vec: 
            str_ ="%s %0.3f"%(str_,val)
        return str_

    def prepareData(self, pd):

        if pd <=0: return 0  

        D = defaultdict(int)
        sentfiletoWrite = open("%s.txt"%(self.sentsFile),"w")
        Logger.logr.info ("Preparing Data for SDAE")
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
                        content = gensim.utils.to_unicode(sentence)
                        content = self.utFunction.normalizeText(content, remove_stopwords=0)
                        sentfiletoWrite.write("%s%s"%(' '.join(content), os.linesep))

        sentfiletoWrite.flush()
        sentfiletoWrite.close()

        oFD = sorted_x = sorted(D.items(), key=operator.itemgetter(1),\
             reverse=True)
        for ii,(w,f) in enumerate(oFD):
            D[w] = ii
        
        self.save_dictionary (self.dictionary, D)


    def runSAE (self):
        """
        Now, it trains the model for 500 epochs.
        """
        trainerr, validerr, testerr = train(saveto = self.saveto,
                                        reload_= self.reload_,
                                        corruption = self.corruption,
                                        corruption_prob = self.corruption_prob,
                                        dim_word = self.dim_word,
                                        dim = self.dim,
                                        n_words = self.n_words,
                                        decay_c = self.decay_c,
                                        param_noise = self.param_noise,
                                        lrate = self.lrate,
                                        optimizer = self.optimizer, 
                                        maxlen = self.maxlen, 
                                        batch_size = self.batch_size,
                                        dictionary = self.dictionary,
                                        valid_batch_size = self.valid_batch_size,
                                        validFreq = 1000,
                                        dispFreq = 10,
                                        saveFreq = 1000,
                                        clip_c = self.clip_c,
                                        encoder = self.encoder,
                                        decoder = self.decoder,
                                        use_preemb = self.use_preemb, 
                                        dataset = self.dataset, 
                                        data_path = "%s.txt"%(self.sentsFile),
                                        use_dropout= self.use_dropout,
                                        embeddings= self.embeddings, 
                                        valid_text = self.valid_text, 
                                        test_text = self.test_text) 

        # The following line is ommitted, as the code does not estimate 
        # the training error and always shows zero at the end
        # Logger.logr.info("train error = %0.4f"%trainerr)


    def runTheBaseline(self, rbase, latent_space_size):
        self.runSAE()
        model = load_model (self.saveto)
        #return 0 

        nSent = 0
        for result in self.postgresConnection.memoryEfficientSelect(["count(*)"],\
            ['sentence'], [], [], []):
            nSent = int (result[0][0])
        sent2vecFileRaw = open("%s_raw"%(self.outembFile),"w") 
        sent2vecFileRaw.write("%s %s%s"%(str(nSent), str(latent_space_size*2), os.linesep))

        sent2vecFile = open("%s.p"%(self.outembFile),"wb")
        sent2vec_dict = {}

        sent2vecFile_raw = open("%s_raw.p"%(self.outembFile),"wb")
        sent2vec_raw_dict = {}

        sentence_list = []
        for result in self.postgresConnection.memoryEfficientSelect(["id", "content"],\
         ["sentence"], [], [], ["id"]):
            for row_id in range(0,len(result)):
                id_ = result[row_id][0] 
                sentence = result[row_id][1]
                content = gensim.utils.to_unicode(sentence) 
                content = self.utFunction.normalizeTextNoStemming(content,\
                    remove_stopwords=0)
                sentence = ' '.join(content)
                sentence_list.append(sentence)

        Logger.logr.info("Total Number of sentences = %i"%len(sentence_list))

        feature_map = encode (model, sentence_list)

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
        outFile = os.path.join(self.dataDir,\
            "%s_repr"%self.latReprName)
        vecFile = open("%s.p"%(outFile),"rb")
        vDict = pickle.load (vecFile)

        summGen = SummaryGenerator (diverse_summ=diversity,\
             postgres_connection = self.postgresConnection,\
             lambda_val = lambda_val)

        summGen.populateSummary(methodId, vDict)

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
        """
        Here, we destroy the database connection.
        """
        self.postgresConnection.disconnectDatabase()