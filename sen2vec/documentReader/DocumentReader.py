#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os 
import sys 
import nltk
import pickle
import gensim

import datetime

from nltk.tokenize import sent_tokenize
from utility.Utility import Utility
from abc import ABCMeta, abstractmethod
from log_manager.log_config import Logger 

from baselineEvaluator.P2VSENTCExecutableEvaluator import P2VSENTCExecutableEvaluator
from baselineEvaluator.FastSentFHVersionEvaluator import FastSentFHVersionEvalutor

from baselineEvaluator.Node2VecEvaluator import Node2VecEvaluator
from baselineEvaluator.TFIDFBaselineEvaluator import TFIDFBaselineEvaluator
from baselineEvaluator.WordVectorAveragingEvaluator import WordVectorAveragingEvaluator

from baselineEvaluator.IterativeUpdatedRetrofitEvaluator import IterativeUpdatedRetrofitEvaluator
from baselineEvaluator.SeqItUpdateEvaluator import SeqItUpdateEvaluator

from baselineEvaluator.RegularizedSen2VecEvaluator import RegularizedSen2VecEvaluator
from baselineEvaluator.SeqRegSentEvaluator import SeqRegSentEvaluator

from baselineEvaluator.JointLearningSen2VecEvaluator import JointLearningSen2VecEvaluator
from baselineEvaluator.FastSentVariantEvaluator import FastSentVariantEvaluator


class DocumentReader:
    """
    DocumentReader Base
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        self.utFunction = Utility("Text Utility")
        self.dataDir = os.environ['TRTESTFOLDER']

    @abstractmethod
    def readDocument(self):
        pass

    """
    Protected Methods
    """
    def _splitIntoParagraphs(self, document):
        """
        This is a rough heuristics. 
        """
        return document.split("%s%s" %(os.linesep, os.linesep))

    def _splitIntoSentences(self, paragraph):
        """
        """
        return sent_tokenize(paragraph)

    def _splitIntoWords(self, sentence):
        pass

    def _folderISHidden(self, folder):
        """
        http://stackoverflow.com/questions/7099290/how-to-ignore-hidden-files-using-os-listdir-python
        """
        return folder.startswith('.') #linux-osx


    def _recordParagraphAndSentence(self, document_id, doc_content, recorder, topic, istrain, skip_short=False):
        """
        It seems Mikolov and others did n't remove the stop words. So, we also do the 
        same for vector construction. 
        """
        paragraphs = self._splitIntoParagraphs(doc_content)

        for position, paragraph in enumerate(paragraphs):
            paragraph_id = self.postgres_recorder.insertIntoParTable(paragraph)
            recorder.insertIntoDocParTable(document_id, paragraph_id, position)
            
            sentences = self._splitIntoSentences(paragraph)
            for sentence_position, sentence in enumerate(sentences):
                sentence_ = self.utFunction.normalizeText(sentence, 0)
                if len(sentence_) <=2:
                    continue 
                elif len(sentence_) < 4 and skip_short == True:
                    continue

                sentence = sentence.replace("\x03","")
                sentence = sentence.replace("\x02","")
                sentence = sentence.replace('\n', ' ').replace('\r', '').strip()
                sentence_id = recorder.insertIntoSenTable(sentence,\
                     topic, istrain, document_id, paragraph_id)
                recorder.insertIntoParSenTable(paragraph_id, sentence_id,\
                    sentence_position)

    def _getTextFromFile(self, file):
        """
        http://stackoverflow.com/questions/7409780/reading-entire-file-in-python
        """ 
        with open(file, encoding='utf-8', errors='ignore') as f:
            return f.read()

    def _getTopics(self, rootDir):
        Logger.logr.info("Starting Reading Topic")
        topic_names, categories = [], []
        for dirName, subdirList, fileList in os.walk(rootDir):
            for topics in subdirList:
                topic_names.append(topics)
                categories.append(topics.split('.')[0])
        self.postgres_recorder.insertIntoTopTable(topic_names, categories)              
        Logger.logr.info("[%i] Topic reading complete." %(len(topic_names)))
        return topic_names



    def performValidation(self, valid_for):
        # Load the optPDict if it is there
        optPDict = {}  
        os.environ['VALID_FOR'] = valid_for 
        os.environ['EVAL'] = 'VALID' 
        dataset_name = os.environ['DATASET']
        latent_space_size = 300


        dict_file = os.path.join(self.dataDir,"%s_optPDict_%s.p"%(os.environ['DATASET'], valid_for))
        dict_file_read, dict_file_write = '', ''

        try:
            dict_file_read = open(dict_file, "rb")
            optPDict = pickle.load (dict_file_read)
            Logger.logr.info("Loaded OptPDict")
        except Exception as e:
            Logger.logr.info (e)
            Logger.logr.info("Creating new OptPDict file")
            
        Logger.logr.info (optPDict)
    
        with open('%s%s%s%s' %(os.environ["TRTESTFOLDER"],"/",dataset_name,\
                "_param_%s_AT_%s.txt"%(valid_for, datetime.datetime.now().isoformat())), 'w') as f:

            paraeval   = P2VSENTCExecutableEvaluator (self.dbstring)
            optPDict   = paraeval.getOptimumParameters (f, optPDict, latent_space_size)

            fheval     = FastSentFHVersionEvalutor (self.dbstring)
            optPDict   = fheval.getOptimumParameters (f, optPDict, latent_space_size)

            wvgeval    = WordVectorAveragingEvaluator (self.dbstring)
            optPDict   = wvgeval.getOptimumParameters (f, optPDict, latent_space_size)

            n2veval    =  Node2VecEvaluator (self.dbstring)
            optPDict   =  n2veval.getOptimumParameters (f, optPDict, latent_space_size)

            regeval    = RegularizedSen2VecEvaluator(self.dbstring)
            optPDict   = regeval.getOptimumParameters (f, optPDict, latent_space_size)

            seqregeval = SeqRegSentEvaluator (self.dbstring)
            optPDict   = seqregeval.getOptimumParameters(f, optPDict, latent_space_size)

            jnteval    = JointLearningSen2VecEvaluator (self.dbstring)
            optPDict   = jnteval.getOptimumParameters(f, optPDict, latent_space_size)

            fstvar     = FastSentVariantEvaluator (self.dbstring)
            optPDict   = fstvar.getOptimumParameters(f, optPDict, latent_space_size)

        # save optDict 
        Logger.logr.info (optPDict)
        dict_file_write = open(dict_file, "wb")
        pickle.dump (optPDict, dict_file_write)


    def performTesting(self, test_for, nIter):
        os.environ["EVAL"]='TEST'
        os.environ['TEST_FOR'] = test_for
        dataset_name = os.environ['DATASET']
        pd, rbase, latent_space_size = 1, 1, 300

        dict_file = os.path.join(self.dataDir,"%s_optPDict_%s.p"%(os.environ['DATASET'], test_for))
        dict_file_read = open(dict_file, "rb")
        optPDict = pickle.load (dict_file_read)
        Logger.logr.info ("Running with following Opt Dictionary settings:-")
        Logger.logr.info (optPDict)

        f = open('%s%s%s%s' %(os.environ["TRTESTFOLDER"],"/",dataset_name,\
                "testresults_%s_AT_%s.txt"%(os.environ['TEST_FOR'],\
                     datetime.datetime.now().isoformat())), 'w') 
        
        niter = nIter
        

        for i in range(0,niter):
            system_list = []
            f.write("###### Iteration: %s ######%s" %(i, os.linesep))

            if test_for == 'RANK':
               self.postgres_recorder.truncateSummaryTable()

            paraeval  = P2VSENTCExecutableEvaluator (self.dbstring)
            paraeval.evaluateOptimum (pd, rbase, latent_space_size, optPDict, f)
            system_list.extend(paraeval.system_id_list)

            fheval    = FastSentFHVersionEvalutor (self.dbstring)
            fheval.evaluateOptimum (pd, rbase, latent_space_size, optPDict, f)
            system_list.extend(fheval.system_id_list)


            tfidfeval =  TFIDFBaselineEvaluator (self.dbstring)
            tfidfeval.evaluateOptimum (pd, rbase, latent_space_size, optPDict, f)
            system_list.extend(tfidfeval.system_id_list)


            wvgeval   = WordVectorAveragingEvaluator (self.dbstring)
            wvgeval.evaluateOptimum (pd, rbase, latent_space_size, optPDict, f)
            system_list.extend(wvgeval.system_id_list)


            itrunner  = IterativeUpdatedRetrofitEvaluator(self.dbstring)
            itrunner.evaluateOptimum(pd, rbase, latent_space_size, optPDict, f)
            system_list.extend(itrunner.system_id_list)


            seqiteval = SeqItUpdateEvaluator (self.dbstring)
            seqiteval.evaluateOptimum (pd, rbase, latent_space_size, optPDict, f)
            system_list.extend(seqiteval.system_id_list)

            regeval   = RegularizedSen2VecEvaluator(self.dbstring)
            regeval.evaluateOptimum (pd, rbase, latent_space_size, optPDict, f)
            system_list.extend(regeval.system_id_list)


            seqregeval = SeqRegSentEvaluator (self.dbstring)
            seqregeval.evaluateOptimum (pd, rbase, latent_space_size, optPDict, f)
            system_list.extend(seqregeval.system_id_list)


            jnteval    = JointLearningSen2VecEvaluator(self.dbstring)
            jnteval.evaluateOptimum (pd, rbase, latent_space_size, optPDict, f)
            system_list.extend(jnteval.system_id_list)


            fstvar     = FastSentVariantEvaluator (self.dbstring)
            fstvar.evaluateOptimum (pd, rbase, latent_space_size, optPDict, f)
            system_list.extend(fstvar.system_id_list)


            if test_for == 'RANK':
               self.runCombinedEvaluation(system_list)
               f.write ("%s%s"%("##Running for Test (100) ######", os.linesep))
               file_name_prefix = "20__"
    
               for system_id in system_list:
                   file_name_prefix = "%s%s_"%(file_name_prefix, str(system_id))

               file_name = "%soutput_100.txt"%file_name_prefix

               file_ = os.path.join(os.environ["SUMMARYFOLDER"], file_name)
               for line in open(file_):
                   f.write(line)
               f.flush()