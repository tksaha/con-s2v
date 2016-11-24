#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os 
import logging 
import operator
from documentReader.DocumentReader import DocumentReader
from documentReader.PostgresDataRecorder   import PostgresDataRecorder
from bs4 import BeautifulSoup
from log_manager.log_config import Logger 
from baselineRunner.Node2VecRunner import Node2VecRunner
from baselineRunner.IterativeUpdateRetrofitRunner import IterativeUpdateRetrofitRunner
from baselineRunner.P2VSENTCExecutableRunner import P2VSENTCExecutableRunner
from baselineRunner.RegularizedSen2VecRunner import RegularizedSen2VecRunner
from baselineRunner.DictRegularizedSen2VecRunner import DictRegularizedSen2VecRunner
from evaluation.rankingevaluation.RankingEvaluation import RankingEvaluation 
from baselineRunner.JointLearningSen2VecRunner import JointLearningSen2VecRunner
from rouge.Rouge import Rouge 

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
        self.lambda_val = os.environ['DUC_LAMBDA']
        self.diversity = os.environ['DUC_DIVERSITY']
        self.duc_topic = os.environ['DUC_TOPIC']
        self.document_id = 0

    def readTopic(self):
        """
        Recording DUC years as topics.
        """
        topic_names = ['2001', '2002', '2003', '2004', '2005', '2006', '2007']
        categories = topic_names
        self.postgres_recorder.insertIntoTopTable(topic_names, categories)
        Logger.logr.info("Topic reading complete.")
    
    
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

    def __readDUC2001(self):
        """
        It loads the DUC 2001 documents into
        the database. Check whether the number of words 
        in the summary is less than 100, if yes then discard. 
        As a rough heuristic, split the sentence and 
        then count number of words. The function also makes 
        sure that there will be no document without summary. 
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
        
        
    def __readDUC2002(self):
        """
        It loads the DUC 2002 documents into
        the database. Check whether the number of words 
        in the summary is less than 100, if yes then discard. 
        As a rough heuristic, split the sentence and 
        then count. The function also makes sure there will 
        be no document without summary. 
        """
        topic = "2002"
        cur_path = "%s/%s" %(self.folderPath, "DUC2002")

        # Go one pass to collect all valid summary file names
        summaries, documents =[], [] 
        for root, directories, files in os.walk(cur_path):
            documents += [os.path.join(root, file_) \
                for file_ in files if file_ not in  ['10', '50', '100', '200', '400', '200e', '400e', 'perdocs']]
            summaries += [os.path.join(root, file_)\
                for file_ in files if file_ in "perdocs"]
 
        summaryFileDict = {}
        summaryFileDict = self.__getValidSummaryFiles(summaries, summaryFileDict)
        Logger.logr.info("Got %i documents and %i summaries"%(len(documents), len(summaryFileDict)))
        
        Logger.logr.info("Recording DUC 2002 Documents.")
        docFileDict = self.recordDocuments(documents, topic, summaryFileDict)
        Logger.logr.info("%i elements in summary dict and %i"\
         " elements in doc dict"%(len(summaryFileDict), len(docFileDict)))
        Logger.logr.info("Recording DUC 2002 Summaries.")
        self.__recordSummariesA(summaries, docFileDict)
        
        
    def readDocument(self, ld): 
        if ld <= 0: return 0 
        self.postgres_recorder.trucateTables()
        self.postgres_recorder.truncateSummaryTable()
        self.postgres_recorder.alterSequences()
        self.readTopic()
        
        document_id = 0
        if self.duc_topic == str(2001):
            self.__readDUC2001()
        else:
            self.__readDUC2002()
        # document_id = self._readDUC2003(document_id)
        # document_id = self._readDUC2004(document_id)
        # document_id = self._readDUC2005(document_id)
        # document_id = self._readDUC2006(document_id)
        # document_id = self._readDUC2007(document_id)


    def __runSpecificEvaluation(self, models = [20], systems = []):
        rougeInstance = Rouge()
        rPDict = rougeInstance.buildRougeParamDict()
        rPDict['-l'] = str(100)
        rPDict['-c'] = str(0.99)

        evaluation = RankingEvaluation(topics = [self.duc_topic], models = models, systems = systems)
        evaluation._prepareFiles()
        evaluation._getRankingEvaluation(rPDict, rougeInstance)

        rPDict['-l'] = str(10)
        evaluation._getRankingEvaluation(rPDict, rougeInstance)
    
    
    def __runCombinedEvaluation(self,system_list):
        rougeInstance = Rouge()
        rPDict = rougeInstance.buildRougeParamDict()
        rPDict['-l'] = str(100)
        rPDict['-c'] = str(0.99)

        evaluation = RankingEvaluation(topics = [self.duc_topic], models = [20], systems = system_list)
        evaluation._prepareFiles()
        evaluation._getRankingEvaluation(rPDict, rougeInstance)

        rPDict['-l'] = str(10)
        evaluation._getRankingEvaluation(rPDict, rougeInstance)
        
        
    def __getRecall(self, method_id, models, systems):
        output_file_name = ""
        for model in models:
            output_file_name += str(model)+"_"
        for system in systems:
            output_file_name += "_"+str(system)
        output_file_name += "_output"
        output_file_name += "_%s.txt" %(str(10))
        
        with open('%s%s%s' %(os.environ["SUMMARYFOLDER"],"/",output_file_name), 'r') as f:
            content = f.read()
            recall = float(content.split("%s ROUGE-1 Average_R: " %method_id)[1].split(' ')[0])
        return recall
    
    
    def runBaselines(self, pd, rbase, gs):
        """
        """

############# Validation ############################       
        with open('%s%s%s%s' %(os.environ["TRTESTFOLDER"],"/",self.duc_topic,"_hyperparameters.txt"), 'w') as f:
            
            latent_space_size = 300

            diversity = False
            if self.diversity == str(1):
                diversity = True 

            # createValidationSet() Need to implement this function
            os.environ['DUC_EVAL']='VALID'
    
            recalls = {}
            window_opt = None #var for the optimal window
            for window in ["8", "10", "12"]:
            #for window in ["8"]:
                Logger.logr.info("Starting Running Para2vec Baseline for Window = %s" %window)
                self.postgres_recorder.truncateSummaryTable()
                paraBaseline = P2VSENTCExecutableRunner(self.dbstring)
                if  window=="8":  
                    paraBaseline.prepareData(pd)
                paraBaseline.runTheBaseline(rbase,latent_space_size, window)
                paraBaseline.generateSummary(gs,\
                    lambda_val=self.lambda_val, diversity=diversity)
                paraBaseline.doHouseKeeping()           
                self.__runSpecificEvaluation(models = [20], systems = [2]) #Running Rouge for method_id = 2 only
                recalls[window] = self.__getRecall(method_id=2, models = [20], systems = [2])
                Logger.logr.info("Recall for %s = %s" %(window, recalls[window]))
            window_opt = max(recalls, key=recalls.get) #get the window for the max recall
            f.write("Optimal window size is %s%s"%(window_opt, os.linesep))
            f.write("P2V Window Recalls: %s%s" %(recalls, os.linesep))
            f.flush()

            Logger.logr.info("Starting Running Para2vec Baseline for Optimal Window = %s" %window_opt)
            self.postgres_recorder.truncateSummaryTable()
            paraBaseline = P2VSENTCExecutableRunner(self.dbstring)
            paraBaseline.runTheBaseline(rbase,latent_space_size, window_opt) #we need the p2v vectors created with optimal window
            paraBaseline.doHouseKeeping()

#           n2vBaseline = Node2VecRunner(self.dbstring)
#           n2vBaseline.prepareData(pd)
#           generate_walk = True 
#           n2vBaseline.runTheBaseline(rbase, latent_space_size, generate_walk)
#           n2vBaseline.doHouseKeeping()

            recalls = {}
            joint_beta_opt = None 
            lambda_list = [0.3, 0.5, 0.8, 1.0]
            method_id = 14
                
            for lambda_ in  lambda_list:
                Logger.logr.info("Starting running jl with lambda = %s" %(lambda_))
                self.postgres_recorder.truncateSummaryTable()
                os.environ["NBR_TYPE"]=str(0)
                os.environ["FULL_DATA"]=str(1)
                os.environ["LAMBDA"]=str(lambda_)
                jointL = JointLearningSen2VecRunner(self.dbstring)
                jointL.window = window_opt
                if lambda_==lambda_list[0]:
                    jointL.prepareData(pd)
                jointL.runTheBaseline(rbase, latent_space_size)
                jointL.generateSummary(gs,method_id,"",\
                        lambda_val=self.lambda_val, diversity=diversity)
                jointL.doHouseKeeping()
                self.__runSpecificEvaluation(models = [20], systems = [method_id]) #Running Rouge for method_id = 2 only
                recalls[lambda_] = self.__getRecall(method_id=method_id, models = [20], systems = [method_id])
                Logger.logr.info("Recall for %s = %s" %(lambda_, recalls[lambda_]))
            joint_beta_opt_full_fixed = max(recalls, key=recalls.get) #get the window for the max recall
            f.write("Optimal lambda is %s%s"%(joint_beta_opt_full_fixed, os.linesep))
            f.write("Recalls joint_beta_opt_full_fixed: %s%s" %(recalls, os.linesep))
            f.flush()


            recalls = {}
            joint_beta_opt = None 
            lambda_list = [0.3, 0.5, 0.8, 1.0]
            method_id = 15
                
            for lambda_ in  lambda_list:
                Logger.logr.info("Starting running jl with lambda = %s" %(lambda_))
                self.postgres_recorder.truncateSummaryTable()
                os.environ["NBR_TYPE"]=str(1)
                os.environ["FULL_DATA"]=str(1)
                os.environ["LAMBDA"]=str(lambda_)
                jointL = JointLearningSen2VecRunner(self.dbstring)
                jointL.window = window_opt
                if lambda_==lambda_list[0]:
                    jointL.prepareData(pd)
                jointL.runTheBaseline(rbase, latent_space_size)
                jointL.generateSummary(gs,method_id,"",\
                        lambda_val=self.lambda_val, diversity=diversity)
                jointL.doHouseKeeping()
                self.__runSpecificEvaluation(models = [20], systems = [method_id]) 
                recalls[lambda_] = self.__getRecall(method_id=method_id, models = [20], systems = [method_id])
                Logger.logr.info("Recall for %s = %s" %(lambda_, recalls[lambda_]))
            joint_beta_opt_full_n2v = max(recalls, key=recalls.get) 
            f.write("Optimal lambda is %s%s"%(joint_beta_opt_full_n2v, os.linesep))
            f.write("Recalls joint_beta_opt_full_n2v: %s%s" %(recalls, os.linesep))
            f.flush()

            recalls = {}
            joint_beta_opt = None 
            lambda_list = [0.3, 0.5, 0.8, 1.0]
            method_id = 16
                
            for lambda_ in  lambda_list:
                Logger.logr.info("Starting running jl with lambda = %s" %(lambda_))
                self.postgres_recorder.truncateSummaryTable()
                os.environ["NBR_TYPE"]=str(0)
                os.environ["FULL_DATA"]=str(0)
                os.environ["LAMBDA"]=str(lambda_)
                jointL = JointLearningSen2VecRunner(self.dbstring)
                jointL.window = window_opt
                if lambda_==lambda_list[0]:
                    jointL.prepareData(pd)
                jointL.runTheBaseline(rbase, latent_space_size)
                jointL.generateSummary(gs,method_id,"",\
                        lambda_val=self.lambda_val, diversity=diversity)
                jointL.doHouseKeeping()
                self.__runSpecificEvaluation(models = [20], systems = [method_id]) 
                recalls[lambda_] = self.__getRecall(method_id=method_id, models = [20], systems = [method_id])
                Logger.logr.info("Recall for %s = %s" %(lambda_, recalls[lambda_]))
            joint_beta_opt_random_fixed = max(recalls, key=recalls.get) 
            f.write("Optimal lambda is %s%s"%(joint_beta_opt_random_fixed, os.linesep))
            f.write("Recalls joint_beta_opt_random_fixed: %s%s" %(recalls, os.linesep))
            f.flush()

            recalls = {}
            joint_beta_opt = None 
            lambda_list = [0.3, 0.5, 0.8, 1.0]
            method_id = 17
                
            for lambda_ in  lambda_list:
                Logger.logr.info("Starting running jl with lambda = %s" %(lambda_))
                self.postgres_recorder.truncateSummaryTable()
                os.environ["NBR_TYPE"]=str(1)
                os.environ["FULL_DATA"]=str(0)
                os.environ["LAMBDA"]=str(lambda_)
                jointL = JointLearningSen2VecRunner(self.dbstring)
                jointL.window = window_opt
                if lambda_==lambda_list[0]:
                    jointL.prepareData(pd)
                jointL.runTheBaseline(rbase, latent_space_size)
                jointL.generateSummary(gs,method_id,"",\
                        lambda_val=self.lambda_val, diversity=diversity)
                jointL.doHouseKeeping()
                self.__runSpecificEvaluation(models = [20], systems = [method_id]) 
                recalls[lambda_] = self.__getRecall(method_id=method_id, models = [20], systems = [method_id])
                Logger.logr.info("Recall for %s = %s" %(lambda_, recalls[lambda_]))
            joint_beta_opt_random_n2v = max(recalls, key=recalls.get) 
            f.write("Optimal lambda is %s%s"%(joint_beta_opt_random_n2v, os.linesep))
            f.write("Recalls joint_beta_opt_random_n2v: %s%s" %(recalls, os.linesep))
            f.flush()
    
            
# ######## Test ########################################
            os.environ["DUC_EVAL"]='TEST'
            system_list = [2]
            for system_id in range(14,18):
                system_list.append(system_id)
            niter = 5
            for i in range(0,niter):
                f.write("###### Iteration: %s ######%s" %(i, os.linesep))
                f.write("Optimal Window: %s%s" %(window_opt, os.linesep))           
                self.postgres_recorder.truncateSummaryTable()
                paraBaseline = P2VSENTCExecutableRunner(self.dbstring)
                paraBaseline.runTheBaseline(rbase,latent_space_size, window_opt) #we need the p2v vectors created with optimal window
                paraBaseline.generateSummary(gs,\
                        lambda_val=self.lambda_val, diversity=diversity)
                paraBaseline.doHouseKeeping()
                f.flush()

                method_id = 14
                os.environ["NBR_TYPE"]=str(0)
                os.environ["FULL_DATA"]=str(1)
                os.environ["LAMBDA"]=str(joint_beta_opt_full_fixed)
                jointL = JointLearningSen2VecRunner(self.dbstring)
                jointL.window = window_opt
                jointL.prepareData(pd)
                jointL.runTheBaseline(rbase, latent_space_size)
                jointL.generateSummary(gs,method_id,"",\
                        lambda_val=self.lambda_val, diversity=diversity)
                jointL.doHouseKeeping()

                method_id = 15
                os.environ["NBR_TYPE"]=str(1)
                os.environ["FULL_DATA"]=str(1)
                os.environ["LAMBDA"]=str(joint_beta_opt_full_n2v)
                jointL = JointLearningSen2VecRunner(self.dbstring)
                jointL.window = window_opt
                jointL.prepareData(pd)
                jointL.runTheBaseline(rbase, latent_space_size)
                jointL.generateSummary(gs,method_id,"",\
                        lambda_val=self.lambda_val, diversity=diversity)
                jointL.doHouseKeeping()

                method_id = 16 
                os.environ["NBR_TYPE"]=str(0)
                os.environ["FULL_DATA"]=str(0)
                os.environ["LAMBDA"]=str(joint_beta_opt_random_fixed)
                jointL = JointLearningSen2VecRunner(self.dbstring)
                jointL.window = window_opt
                jointL.prepareData(pd)
                jointL.runTheBaseline(rbase, latent_space_size)
                jointL.generateSummary(gs,method_id,"",\
                        lambda_val=self.lambda_val, diversity=diversity)
                jointL.doHouseKeeping()

                method_id = 17
                os.environ["NBR_TYPE"]=str(1)
                os.environ["FULL_DATA"]=str(0)
                os.environ["LAMBDA"]=str(joint_beta_opt_random_n2v)
                jointL = JointLearningSen2VecRunner(self.dbstring)
                jointL.window = window_opt
                jointL.prepareData(pd)
                jointL.runTheBaseline(rbase, latent_space_size)
                jointL.generateSummary(gs,method_id,"",\
                        lambda_val=self.lambda_val, diversity=diversity)
                jointL.doHouseKeeping()
                self.__runCombinedEvaluation(system_list)
                
                f.write ("%s%s"%("#########################Running for Test (100) ###########################################", os.linesep))
                file_name_prefix = "20__"
            
                for system_id in system_list:
                    file_name_prefix = "%s%s_"%(file_name_prefix, str(system_id))

                file_name = "%soutput_100.txt"%file_name_prefix

                file_ = os.path.join(os.environ["SUMMARYFOLDER"],file_name)
                for line in open(file_):
                    f.write(line)
                f.flush()

                f.write ("%s%s"%("#########################Running for Test (10) ###########################################", os.linesep))
                #file_name = "20__14_15_16_17_output_10.txt"
                file_name = "%soutput_10.txt"%file_name_prefix
                file_ = os.path.join(os.environ["SUMMARYFOLDER"], file_name)
                for line in open(file_):
                    f.write(line)

                f.write("%s%s"%(os.linesep, os.linesep))
                f.flush()




