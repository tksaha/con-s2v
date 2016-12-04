#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os 
import logging 
import re
import numpy as np
from documentReader.DocumentReader import DocumentReader
from documentReader.PostgresDataRecorder   import PostgresDataRecorder
from log_manager.log_config import Logger
from baselineRunner.Paragraph2VecSentenceRunner  import Paragraph2VecSentenceRunner
from baselineRunner.Node2VecRunner  import Node2VecRunner
from baselineRunner.Paragraph2VecRunner import Paragraph2VecRunner
from baselineRunner.Paragraph2VecCEXERunner import Paragraph2VecCEXERunner
from bs4 import BeautifulSoup


class IMDBReader(DocumentReader):
    """ 
    IMDB Document Reader. Reads IMDB documents extracted from 
    : 
    """

    def __init__(self,*args, **kwargs):
        """
        Initialization assumes that IMDB_PATH environment is set. 
        To set in linux or mac: export IMDB_PATH=/some_directory_containing_IMDB_data
        """
        DocumentReader.__init__(self, *args, **kwargs)
        self.dbstring = os.environ["IMDB_DBSTRING"]
        self.postgres_recorder = PostgresDataRecorder(self.dbstring)
        self.folderPath = os.environ['IMDB_PATH']
        self.validationDict = {}
    


    def insertIntoDatabase(self, sent_, document_id, topic, istrain, metadata):
        self.postgres_recorder.insertIntoDocTable(document_id, 'imdb', \
                                sent_, 'imdb', metadata)
        self.postgres_recorder.insertIntoDocTopTable(document_id,\
                    [topic], [topic]) 
        sentence_id = self.postgres_recorder.insertIntoSenTable(sent_,\
                     topic, istrain, document_id, 1)

    def __readAPass(self, load=0):
        train_doc_ids = []
        topic_names = ['pos', 'neg', 'unsup']

        if load == 0:
            for topic in topic_names:
                self.postgres_recorder.insertIntoTopTable([topic], [topic]) 
        
        document_id = 0
        for first_level_folder in next(os.walk(self.folderPath))[1]:

            if not(DocumentReader._folderISHidden(self, first_level_folder)):
                for topic in topic_names:                   
                    if first_level_folder == 'test' and topic == 'unsup':
                        continue
                    for file_ in os.listdir("%s%s%s%s%s" %(self.folderPath, "/", \
                                            first_level_folder, "/", topic)):
                        doc_content = self._getTextFromFile("%s%s%s%s%s%s%s" \
                            %(self.folderPath, "/", first_level_folder, "/", topic, "/", file_))
                        
                        doc_content = BeautifulSoup(doc_content, "html.parser").text
                        document_id += 1
                        title, metadata, istrain = None, None, None                 
                        try:
                            trainortest = first_level_folder
                            metadata = "SPLIT:%s"%trainortest
                            if trainortest.lower() == 'train' and topic !='unsup':
                               if load ==0:
                                  train_doc_ids.append(document_id)
                               istrain = 'YES'
                            elif trainortest.lower() == 'train' and topic =='unsup':
                               istrain = 'MAYBE'
                            else:
                               istrain = 'NO'
                            if load ==1:
                                if document_id in self.validationDict:
                                    istrain = 'VALID'
                        except Exception as e:
                            Logger.logr.info("NO MetaData or Train Test Tag %s" %e)

                        if load ==1:
                           self.insertIntoDatabase(doc_content, document_id, topic, istrain, metadata)

        return train_doc_ids

    def __createValidationSet(self, document_ids):

        total_doc = len(document_ids)
        nvalid_doc = float(total_doc * 0.20)

        np.random.seed(2000)
        valid_list = np.random.choice(document_ids, nvalid_doc, replace=False).tolist()

        for id_ in valid_list:
            self.validationDict[id_] = 1


    def readDocument(self, ld): 
        """
        """
        if ld <= 0: return 0            
        self.postgres_recorder.trucateTables()
        self.postgres_recorder.alterSequences()

        train_doc_ids = self.__readAPass(0)
        self.__createValidationSet(train_doc_ids)
        self.__readAPass(1)
                    
        Logger.logr.info("Document reading complete.")
        return 1
    
    
    def runBaselines(self, pd, rbase, gs):
        """
        """
        optDict = self._runClassificationOnValidation(pd, rbase, gs,"imdb")
        self.doTesting(optDict, "imdb", rbase, pd, gs, True)
