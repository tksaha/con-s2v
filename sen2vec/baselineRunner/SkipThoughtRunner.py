#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os 
import logging 
import re
import numpy as np 
import gensim 
from skipThought.training import vocab, train, tools 
from baselineRunner.BaselineRunner import BaselineRunner
from log_manager.log_config import Logger 
from utility.Utility import Utility


class SkipThoughtRunner(BaselineRunner):
    def __init__(self, *args, **kwargs):
        """
        """
        BaselineRunner.__init__(self, *args, **kwargs)
        self.latReprName = "skip-thought"
        self.rootdir = os.environ['SEN2VEC_DIR']
        self.postgresConnection.connectDatabase()
        self.utFunction = Utility("Text Utility")
        self.sentIDList = list()
        self.sentenceList = list()
        self.dataDir = os.environ['TRTESTFOLDER']
        self.system_id = 87


    def prepareData(self, pd):
        """
        Suppose that you have a list of strings available for training, 
        where the contents of the entries are contiguous 
        (so the (i+1)th entry is the sentence that follows 
        the i-th entry. As an example, you can download our 
        BookCorpus dataset, which was used for training the 
        models available on the main page. Lets call this list X. 
        Note that each string should already be tokenized 
        (so that split() will return the desired tokens).
        """

        Logger.logr.info ("Preparing Data for Skip-Thought")
        for doc_result in self.postgresConnection.memoryEfficientSelect(["id"],\
            ["document"], [], [], ["id"]):
            for row_id in range(0,len(doc_result)):
                document_id = doc_result[row_id][0]
                #Logger.logr.info("Working for Document id =%i", doc_result[row_id][0])
                for sentence_result in self.postgresConnection.memoryEfficientSelect(\
                    ['id','content'],['sentence'],[["doc_id","=",document_id]],[],['id']):
                    for inrow_id in range(0, len(sentence_result)):
                        sentence_id = int(sentence_result[inrow_id][0])
                        sentence = sentence_result[inrow_id][1]
                        content = gensim.utils.to_unicode(sentence) 
                        content = self.utFunction.normalizeText(content, remove_stopwords=0)
                        self.sentenceList.append(' '.join(content))
                        self.sentIDList.append(sentence_id)
           

        if  pd >0:    
            loc = os.path.join(self.dataDir, "dictionary.p" )
            worddict, wordcount = vocab.build_dictionary (self.sentenceList)
            vocab.save_dictionary (worddict, wordcount, loc)

    def runTheBaseline(self, rbase, latent_space_size):

        if rbase <=0: return 0 

        Logger.logr.info ("Running The Baseline ")
        print (len(self.sentenceList))

        train.trainer(self.sentenceList, 
            dim_word=latent_space_size, # word vector dimensionality
            dim=latent_space_size*2, # the number of GRU units
            encoder='gru',
            decoder='gru',
            max_epochs=5,
            dispFreq=1,
            decay_c=0.,
            grad_clip=5.,
            n_words=20000, # This is the most important parameter
            maxlen_w=1000,
            optimizer='adam',
            batch_size = 64,
            saveto= os.path.join(self.dataDir, "model.npz"),
            dictionary= os.path.join(self.dataDir, "dictionary.p"),
            saveFreq=100,
            reload_= True)
        
        #model = tools.load_model(embed_map)

    def runEvaluationTask(self):
        
        from skipThought.training import tools 
        embed_map = {}
        model = tools.load_model("../Data/model_news.npz", "../Data/dictionary.p", embed_map); 
        model.encode(X)

    def doHouseKeeping(self):
        pass 