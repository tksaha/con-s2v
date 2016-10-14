#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os 
import logging 
import operator
from documentReader.DocumentReader import DocumentReader
from documentReader.PostgresDataRecorder   import PostgresDataRecorder
from bs4 import BeautifulSoup
from log_manager.log_config import Logger 
from baselineRunner.Paragraph2VecSentenceRunner  import Paragraph2VecSentenceRunner 
from baselineRunner.Node2VecRunner import Node2VecRunner
from baselineRunner.IterativeUpdateRetrofitRunner import IterativeUpdateRetrofitRunner
from baselineRunner.P2VSENTCExecutableRunner import P2VSENTCExecutableRunner

from evaluation.rankingevaluation.RankingEvaluation import RankingEvaluation 

# There are some summaries [ex:fbis4-45908, FT932-15960] for which the 
# original document is not present
class DUCReader(DocumentReader):
    """ 
    DUC Document Reader

    """

    def __init__(self,*args, **kwargs):
        """
        It reads he environment variable and initializes the 
        base class. 
        """
        DocumentReader.__init__(self, *args, **kwargs)
        self.dbstring = os.environ["DUC_DBSTRING"]
        self.postgres_recorder = PostgresDataRecorder(self.dbstring)
        self.folderPath = os.environ['DUC_PATH']
        self.processed_filenames = []
        self.processed_summaries = []
        self.document_id = 0

    def readTopic(self):
        """
        Recording DUC years as topics.
        """
        topic_names = ['2001', '2002', '2003', '2004', '2005', '2006', '2007']
        categories = topic_names
        self.postgres_recorder.insertIntoTopTable(topic_names, categories)
        Logger.logr.info("Topic reading complete.")
    
    def recordFirstSentenceBaselineSummary(self, document_id):
        """
        Recording First Sentence as a Baseline Summary
        """
        sentence_id = self.postgres_recorder.selectFirstSentBaselineId(document_id)
        method_id = 21 #First sentence as a baseline summary= 21
        position = 1
        self.postgres_recorder.insertIntoSumTable(document_id, method_id, sentence_id, position)
    
    def recordDocuments(self, documents, topic, summaryFileDict):
        docFileDict = {}

        for document in documents:
            filename = document.split(os.path.sep)[-1] #ft923-5089
            if filename in self.processed_filenames: #don't store duplicate files
                continue
            if filename not in summaryFileDict:
                continue

            doc_content = self._getTextFromFile("%s" %(document))
            soup = BeautifulSoup(doc_content, "html.parser")
        
            try:
                doc_content = soup.find('text').text.strip()
            except:
                Logger.logr.info("%s %s" %(document, "Skipping. Cause, TEXT tag not found"))
                continue
            if doc_content.count('.') > 1000 or doc_content.count('.') < 1:
                Logger.logr.info("%s %s" %(document, "Skipping. Cause, %s sentences." %doc_content.count('.')))
                continue

            if len(doc_content.split()) < 100:
                continue


            self.processed_filenames += [filename]
            docFileDict [filename] = 1
            self.document_id += 1
            title, metadata, istrain = None, None, 'YES'


            self.postgres_recorder.insertIntoDocTable(self.document_id, title, \
                        doc_content, filename, metadata) 
            category = topic.split('.')[0]
            self.postgres_recorder.insertIntoDocTopTable(self.document_id, \
                        [topic], [category])
            self._recordParagraphAndSentence(self.document_id, doc_content, self.postgres_recorder, topic, istrain)
            
            self.recordFirstSentenceBaselineSummary(self.document_id) #Recording First Sentence as a Baseline Summary

        return docFileDict
        
    def __recordSummariesA(self, summaries, document_dict):
        """
        First check whether corresponding valid document is in 
        the database
        """
        for summary in summaries:
            doc_content = self._getTextFromFile("%s" %(summary))
            soup = BeautifulSoup(doc_content, "html.parser")
            sums = soup.findAll('sum')

            for sum_ in sums:
                filename = sum_.get('docref')
                doc_content = sum_.text.strip()
                if filename not in document_dict:
                    Logger.logr.info("Checking %s in document dict"%filename)
                    continue

                method_id = 20 #DUC = 20
                summarizer = sum_.get('summarizer')
                metadata = "SUMMARIZER:%s" %(summarizer)
                if "%s%s" %(filename, summarizer) in self.processed_summaries:
                    continue
                self.processed_summaries += ["%s%s" %(filename, summarizer)]
                self.postgres_recorder.insertIntoGoldSumTable(filename, doc_content, \
                            method_id, metadata)

    def __getSummaryFileNames(self, summaryFile):
        doc_content = self._getTextFromFile("%s" %(summaryFile))
        soup = BeautifulSoup(doc_content, "html.parser")
        summaries = soup.findAll('sum')
        filenames = []

        for summary in summaries:
            filename = summary.get('docref')
            doc_content = summary.text
            if len(doc_content.split()) <100:
                continue
            else:
                filenames.append(filename)

        return filenames

    def __getValidSummaryFiles(self, summaries, summaryFileDict):
        
        for summary in summaries: 
            fileNames = self.__getSummaryFileNames(summary)
            for names in fileNames:
                summaryFileDict[names] = 1
        return summaryFileDict

    def __readDUC2001(self, document_id):
        """
        It loads the DUC 2001 documents into
        the database. Check whether the number of words 
        in the summary is less than 100, if yes then discard. 
        As a rough heuristic, split the sentence and 
        then count. The function also makes sure there will 
        be no document without summary. 
        """
        topic = "2001"
        cur_path = "%s/%s" %(self.folderPath, "DUC2001")

        # Go one pass to collect all valid summary file names
        summaries, documents =[], [] 
        for root, directories, files in os.walk(cur_path):
            documents += [os.path.join(root, file_) \
                for file_ in files if file_ not in  ['50', '100', '200', '400', 'perdocs']]
            summaries += [os.path.join(root, file_)\
                for file_ in files if file_ in "perdocs"]
 
        
        summaryFileDict = {}
        summaryFileDict = self.__getValidSummaryFiles(summaries, summaryFileDict)
        Logger.logr.info("Got %i documents and %i summaries"%(len(documents), len(summaryFileDict)))
        
        Logger.logr.info("Recording DUC 2001 Documents.")
        docFileDict = self.recordDocuments(documents, topic, summaryFileDict)


        Logger.logr.info("%i elements in summary dict and %i"\
         " elements in doc dict"%(len(summaryFileDict), len(docFileDict)))


        Logger.logr.info("Recording DUC 2001 Summaries.")
        self.__recordSummariesA(summaries, docFileDict)
        
        
    
    # def __readDUC2002(self, document_id):
    #   """
    #   It loads the DUC 2002 documents into
    #   the database
    #   """
    #   topic = "2002"
    #   cur_path = "%s/%s/%s" %(self.folderPath, "DUC2002", "summaries")

    #   # Go one pass to collect all valid summary file names 
    #   summaryFileDict = {}
    #   summaryFileDict = self.__getValidSummaryFiles(cur_path, summaryFileDict)

    #   cur_path = "%s/%s/%s" %(self.folderPath, "DUC2002", "docs")
    #   documents = []
    #   for root, directories, files in os.walk(cur_path):
    #       documents += [os.path.join(root, file_) for file_ in files]
        
    #   Logger.logr.info("Recording DUC 2002 Documents.")
    #   document_dict = self.recordDocuments(documents, topic, summaryFileDict)
        
    #   Logger.logr.info("Recording DUC 2001 Summaries.")
    #   self.recordSummariesA(summaries, document_dict)


        
        
        
    # def _readDUC2003(self, document_id):
    #   """
    #   It loads the DUC 2003 documents into
    #   the database
    #   """
    #   topic = "2003"
    #   cur_path = "%s/%s/%s" %(self.folderPath, "DUC2003", "duc2003_testdata")
    #   documents = []

    #   for root, directories, files in os.walk(cur_path):
    #       documents += [os.path.join(root, file_) for file_ in files]
        
    #   Logger.logr.info("Recording DUC 2003 Documents.")
    #   document_id = self.recordDocuments(documents, document_id,topic)
                
    #   cur_path = "%s/%s/%s/%s" %(self.folderPath, "DUC2003", "detagged.duc2003.abstracts", "models")
    #   summaries = []
    #   for root, directories, files in os.walk(cur_path):
    #       summaries += [os.path.join(root, file_) for file_ in files if file_.split('.')[1] == 'P']
        
    #   Logger.logr.info("Recording DUC 2003 Summaries.")
    #   self.recordSummariesB(summaries)
        
    #   return document_id
        
    # def _readDUC2004(self, document_id):
    #   """
    #   It loads the DUC 2004 documents into
    #   the database
    #   """
    #   topic = "2004"
    #   cur_path = "%s/%s/%s" %(self.folderPath, "DUC2004", "duc2004_testdata")
    #   summaries = []
    #   documents = []
    #   for root, directories, files in os.walk(cur_path):
    #       for file_ in files:
    #           if file_ in ['50', '100', '200', '400']:
    #               pass
    #           elif file_ in ['perdocs']:
    #               summaries += [os.path.join(root, file_)]
    #           else:
    #               documents += [os.path.join(root, file_)]
        
    #   Logger.logr.info("Recording DUC 2004 Documents.")
    #   document_id = self.recordDocuments(documents, document_id, topic)
        
    #   cur_path = "%s/%s/%s/%s/%s" %(self.folderPath, "DUC2004", "duc2004_results", "ROUGE", "eval")
    #   summaries = []
    #   for root, directories, files in os.walk(cur_path):
    #       summaries += [os.path.join(root, file_) for file_ in files if file_.split('.')[1] == 'P']
        
    #   Logger.logr.info("Recording DUC 2004 Summaries.")
    #   self.recordSummariesA(summaries)

    #   return document_id
        
    # def _readDUC2005(self, document_id):
    #   """
    #   It loads the DUC 2005 documents into
    #   the database
    #   """
    #   topic = "2005"
    #   cur_path = "%s/%s" %(self.folderPath, "DUC2005")
    #   summaries = []
    #   documents = []
    #   for root, directories, files in os.walk(cur_path):
    #       for file_ in files:
    #           if file_ in ['50', '100', '200', '400']:
    #               pass
    #           elif file_ in ['perdocs']:
    #               summaries += [os.path.join(root, file_)]
    #           else:
    #               documents += [os.path.join(root, file_)]
        
    #   Logger.logr.info("Recording DUC 2005 Documents.")
    #   document_id = self.recordDocuments(documents, document_id, topic)

    #   return document_id
        
    # def _readDUC2006(self, document_id):
    #   """
    #   It loads the DUC 2006 documents into
    #   the database
    #   """
    #   topic = "2006"
    #   cur_path = "%s/%s" %(self.folderPath, "DUC2006")
    #   summaries = []
    #   documents = []
    #   for root, directories, files in os.walk(cur_path):
    #       for file_ in files:
    #           if file_ in ['50', '100', '200', '400']:
    #               pass
    #           elif file_ in ['perdocs']:
    #               summaries += [os.path.join(root, file_)]
    #           else:
    #               documents += [os.path.join(root, file_)]
        
    #   Logger.logr.info("Recording DUC 2006 Documents.")
    #   document_id = self.recordDocuments(documents, document_id, topic)

    #   return document_id
        
    # def _readDUC2007(self, document_id):
    #   """
    #   It loads the DUC 2007 documents into
    #   the database
    #   """
    #   topic = "2007"
    #   cur_path = "%s/%s" %(self.folderPath, "DUC2007")
    #   summaries = []
    #   documents = []
    #   for root, directories, files in os.walk(cur_path):
    #       for file_ in files:
    #           if file_ in ['50', '100', '200', '400']:
    #               pass
    #           elif file_ in ['perdocs']:
    #               summaries += [os.path.join(root, file_)]
    #           else:
    #               documents += [os.path.join(root, file_)]
        
    #   Logger.logr.info("Recording DUC 2007 Documents.")
    #   document_id = self.recordDocuments(documents, document_id, topic)

    #   return document_id
    
    def readDocument(self, ld): 
        if ld <= 0: return 0 
        self.postgres_recorder.trucateTables()
        self.postgres_recorder.truncateSummaryTable()
        self.postgres_recorder.alterSequences()
        self.readTopic()
        
        document_id = 0
        document_id = self.__readDUC2001(document_id)
        # document_id = self._readDUC2002(document_id)
        # document_id = self._readDUC2003(document_id)
        # document_id = self._readDUC2004(document_id)
        # document_id = self._readDUC2005(document_id)
        # document_id = self._readDUC2006(document_id)
        # document_id = self._readDUC2007(document_id)
        
    def runBaselines(self, pd, rbase, gs):
        """
        """
        latent_space_size = 300
    
        # Logger.logr.info("Starting Running Para2vec Baseline")
        # paraBaseline = P2VSENTCExecutableRunner(self.dbstring)
        # paraBaseline.prepareData(pd)
        # paraBaseline.runTheBaseline(rbase,latent_space_size)
        # paraBaseline.generateSummary(gs)
        # paraBaseline.runEvaluationTask()

        # Logger.logr.info("Starting Running Node2vec Baseline")    
        # n2vBaseline = Node2VecRunner(self.dbstring)
        # n2vBaseline.prepareData(pd)
        # n2vBaseline.runTheBaseline(rbase, latent_space_size)
        # n2vBaseline.generateSummary(gs, 3, "")
        # n2vBaseline.generateSummary(gs, 4, "_init")
        # n2vBaseline.generateSummary(gs, 5, "_retrofit")
        # n2vBaseline.runEvaluationTask()

        # iterrunner = IterativeUpdateRetrofitRunner(self.dbstring)
        # iterrunner.prepareData(pd)
        # iterrunner.runTheBaseline(rbase)
        # iterrunner.generateSummary(gs, 6, "_unweighted")
        # iterrunner.generateSummary(gs, 7, "_weighted")
        # iterrunner.runEvaluationTask()

        # evaluation = RankingEvaluation(topics = ['2003'], models = [20], systems = [1, 2, 3, 4, 5, 21])
        # evaluation._getRankingEvaluation()
