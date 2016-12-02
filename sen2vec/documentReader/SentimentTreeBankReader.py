#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os 
import logging 
import re
from documentReader.DocumentReader import DocumentReader
from documentReader.PostgresDataRecorder   import PostgresDataRecorder
from log_manager.log_config import Logger
from baselineRunner.Paragraph2VecSentenceRunner  import Paragraph2VecSentenceRunner
from baselineRunner.Node2VecRunner  import Node2VecRunner
from baselineRunner.Paragraph2VecRunner import Paragraph2VecRunner
from baselineRunner.Paragraph2VecCEXERunner import Paragraph2VecCEXERunner


class SentimentTreeBank2WayReader(DocumentReader):
    def __init__(self, *args, **kwargs):
        """
        Initialization assumes that SENTTREE_PATH environment is set. 
        """
        DocumentReader.__init__(self, *args, **kwargs)
        self.dbstring = os.environ["SENTTREE_DBSTRING"]
        self.postgres_recorder = PostgresDataRecorder(self.dbstring)
        self.folderPath = os.environ['SENTTREE_PATH']

    def readTopic(self):
        topic_names =  ['pos', 'neg','neutral']
        categories  =  ['pos', 'neg', 'neutral']

        self.postgres_recorder.insertIntoTopTable(topic_names, categories)              
        Logger.logr.info("[%i] Topic reading complete." %(len(topic_names)))
        return topic_names

    def readDSplit(self,fileName):
        """
        1 Train, 2 Test, 3 dev
        """
        line_count = 0 
        dSPlitDict = {}
        for line in open(fileName, 'rb'):
            if line_count == 0: 
                pass
            else:   
                doc_id,_, splitid = (line.decode('utf-8').strip()).partition(",")
                dSPlitDict[int(doc_id)] = int(splitid)
            line_count = line_count + 1

        Logger.logr.info("Finished reading %i sentences and their splits"%line_count)

        return dSPlitDict;

    def readSentences(self,fileName):
        line_count = 0
        sentenceDict = {}
        for line in open(fileName, 'rb'):
            if line_count == 0:
                pass
            else:       
                doc_id,_,sentence = (line.decode('utf-8').strip()).partition("\t")
                sentenceDict[int(doc_id)] = sentence.strip()
            line_count = line_count + 1
        return sentenceDict
        Logger.logr.info("Finished reading %i sentence"%line_count)

    def phraseToSentiment(self, fileName):
        line_count = 0 
        phraseToSentimentDict = {}

        for line in open(fileName, 'rb'):
            if line_count == 0:
                pass
            else:
                phrase_id,_, sentiment = (line.decode('utf-8').strip()).partition("|")
                phraseToSentimentDict[int(phrase_id)] = float(sentiment)
            line_count = line_count + 1
        return phraseToSentimentDict
        Logger.logr.info("Finished reading %i phrases"%line_count)

    def getTopicCategory(self, sentiment_val):
        """
        [0, 0.2] very negative 
        (0.2, 0.4] negative 
        (0.4, 0.6] neutral 
        (0.6, 0.8] positive 
        (0.8, 1.0] very positive
        """
        if sentiment_val <= 0.2: 
            return ('vng', 'vng')
        elif sentiment_val > 0.2 and sentiment_val <= 0.4:
            return ('ng', 'ng')
        elif sentiment_val > 0.4 and sentiment_val<= 0.6:
            return ('ntr','ntrl')
        elif sentiment_val > 0.6 and sentiment_val<= 0.8:
            return ('pos', 'pos')
        else:
            return ('vpos', 'vpos')

    def insertIntoDatabase(self, sent_, document_id, topic, istrain, metadata):
        self.postgres_recorder.insertIntoDocTable(document_id, 'senttree', \
                                sent_, 'senttree', metadata)
        self.postgres_recorder.insertIntoDocTopTable(document_id,\
                    [topic], [topic]) 
        sentence_id = self.postgres_recorder.insertIntoSenTable(sent_,\
                     topic, istrain, document_id, 1)

    def readDocument(self, ld): 
        """
        SKip neutral phrases 
        """

        if ld <= 0: return 0            
        self.postgres_recorder.trucateTables()
        self.postgres_recorder.alterSequences()
        topic_names = self.readTopic()

        allPhrasesFile = "%s/dictionary.txt"%(self.folderPath)
        dSPlitDict = self.readDSplit("%s/datasetSplit.txt"%self.folderPath)
        sentenceDict = self.readSentences("%s/datasetSentences.txt"%self.folderPath)
        phraseToSentimentDict = self.phraseToSentiment("%s/sentiment_labels.txt"%self.folderPath)

        sentence_id = 0

        for sent_id, sentence in sentenceDict.items():
            is_a_sentence = False
            sentence_id = sent_id
            sentiment_val = -1

            for line in open(allPhrasesFile,'rb'):
                phrase, _ , phrase_id = (line.decode('utf-8').strip()).partition("|")
                sentence_mod = sentence.replace("-LRB-","(").replace("-RRB-",")")
                if phrase == sentence or phrase==sentence_mod:
                    sentiment_val = phraseToSentimentDict[int(phrase_id)]           
                    topic, category = self.getTopicCategory(sentiment_val)
                    is_a_sentence = True 
                    break 
                    #self.insertIntoDatabase(phrase, sentence_id, topic, istrain, metadata)
        

            
            print ("%s\t%s"%(sent_id,sentiment_val))


           
    
        Logger.logr.info("Document reading complete.")
        return 1

    def runBaselines(self, pd, rbase, gs):
        """
        Discuss with Joty about the clustering settings. 
        """
        optDict = self._runClassificationOnValidation(pd, rbase, gs,"stree")
        self.doTesting(optDict, "stree", rbase, pd, gs, True)


        #optDict = self._runClusteringOnValidation(pd, rbase, gs, "news")
        #self.doTesting(optDict, "news", rbase, pd, gs, False)

        #optDict = self._SuprunClassificationOnValidation(pd, rbase, gs,"news")
        #optDict ={}
        #self.doTesting_Sup(optDict, "news", rbase, pd, gs, True)

        #optDict = self._runFastSentClassificationValidation(pd, rbase, gs, "news")
        #self.doTesting_FastSent(optDict, "news", rbase, pd, gs, True)

        #optDict = self._runFastSentClusteringValidation(pd, rbase, gs, "news")
        #self.doTesting_FastSent(optDict, "news", rbase, pd, gs, False)
        
