#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os 
import logging 
from documentReader.DocumentReader import DocumentReader
from documentReader.PostgresDataRecorder   import PostgresDataRecorder
from log_manager.log_config import Logger
import re
import numpy as np 
import pickle
import operator;
from baselineRunner.Paragraph2VecSentenceRunner  import Paragraph2VecSentenceRunner 
from baselineRunner.Node2VecRunner import Node2VecRunner
from baselineRunner.IterativeUpdateRetrofitRunner import IterativeUpdateRetrofitRunner
from baselineRunner.P2VSENTCExecutableRunner import P2VSENTCExecutableRunner
from baselineRunner.JointSupervisedRunner import JointSupervisedRunner
from baselineRunner.JointLearningSen2VecRunner import JointLearningSen2VecRunner
from baselineRunner.FastSentVariantRunner import FastSentVariantRunner
from evaluation.rankingevaluation.RankingEvaluation import RankingEvaluation

class SICKReader(DocumentReader):
    """ 
    SICK Reader.
    """

    def __init__(self,*args, **kwargs):
        """
        Initialization assumes that NEWSGROUP_PATH environment is set. 
        To set in linux or mac: export NEWSGROUP_PATH=/some_directory_containing_newsgroup_data
        """
        DocumentReader.__init__(self, *args, **kwargs)
        self.dbstring = os.environ["SICK_DBSTRING"]
        self.postgres_recorder = PostgresDataRecorder(self.dbstring)
        self.folderPath = os.environ['SICK_PATH']
        self.rootdir = os.environ['SEN2VEC_DIR']
        self.validation_pairs = {}
        self.test_pairs = {}
        self.train_pairs = {}
        self.sp_methods = {}
        self.pearson_methods = {}


    def insertIntoDatabase(self, sent_, document_id, topic):
        self.postgres_recorder.insertIntoDocTable(document_id, 'SICK', \
                                sent_, 'SICK', topic)
        self.postgres_recorder.insertIntoDocTopTable(document_id,\
                    [topic], [topic]) 
        sentence_id = self.postgres_recorder.insertIntoSenTable(sent_,\
                     topic, topic, document_id, 1)

        return sentence_id

    def readDocument(self, ld): 
        """
        Stripping is by default inactive. For future reference it has been 
        imported from scikit-learn newsgroup reader package. 

        
        """
        if ld <= 0: return 0            
        self.postgres_recorder.trucateTables()
        self.postgres_recorder.alterSequences()

        # Read SICK Data
        sick_file = open(os.path.join(self.folderPath,"SICK.txt"))

        self.postgres_recorder.insertIntoTopTable(['TRAIN'], ['TRAIN'])
        self.postgres_recorder.insertIntoTopTable(['TEST'],['TEST'])
        self.postgres_recorder.insertIntoTopTable(['TRIAL'], ['TRIAL'])

        first_line = True 
        document_id = 1
        for line in sick_file: 
            if first_line:
                first_line =False
            else:
                sentence_pair = line.strip().split("\t")
                sent_1 = sentence_pair[1]
                sent_2 = sentence_pair[2]
                topic = sentence_pair[len(sentence_pair)-1].strip()
                db_id_1 = self.insertIntoDatabase(sent_1, document_id, topic)
                document_id +=1
                db_id_2 = self.insertIntoDatabase(sent_2, document_id, topic)
                document_id +=1
                if  topic == 'TEST':
                    self.test_pairs[(db_id_1, db_id_2)] = float(sentence_pair[4])
                elif  topic == 'TRIAL':
                    self.validation_pairs[(db_id_1,db_id_2)] = float(sentence_pair[4])
                elif  topic == 'TRAIN':
                    self.train_pairs[(db_id_1,db_id_2)] = float(sentence_pair[4])


        # dump pickled dictionary
        test_pair_file =  open(os.path.join(self.rootdir,"Data/test_pair_sick.p"),"wb")
        validation_pair_file = open(os.path.join(self.rootdir,"Data/validation_pair_sick.p"),"wb")
        train_pair_file =  open(os.path.join(self.rootdir,"Data/train_pair_sick.p"),"wb")

        pickle.dump (self.test_pairs, test_pair_file)
        pickle.dump (self.validation_pairs, validation_pair_file)
        pickle.dump (self.train_pairs,train_pair_file)

        return 1
    

    def insertevals (self, sp, pearson, repr_name):
        if  repr_name in self.sp_methods:
            value_list = self.sp_methods[repr_name]
            value_list.append(sp)
            self.sp_methods[repr_name] = value_list
            value_list_p = self.pearson_methods[repr_name]
            value_list_p.append(pearson)
            self.pearson_methods[repr_name] = value_list_p
            
        else:
            sp_list = [sp]
            pearson_list = [pearson]
            self.sp_methods[repr_name] = sp_list
            self.pearson_methods[repr_name] = pearson_list

    
    
    def runBaselines(self, pd, rbase, gs):
        """
        """

        dataset= 'sick'
        os.environ['TEST_AND_TRAIN'] = 'YES'
        latent_space_size = 300

        ############# Validation ############################       
        with open('%s%s%s%s' %(os.environ["TRTESTFOLDER"],"/","sick","_hyperparameters.txt"), 'w') as f:
            os.environ['EVAL']='VALID'
    
            spearman = {}  #spearman or pearson correlation
            window_opt = None #var for the optimal window
            for window in ["8", "10", "12"]:
            #for window in ["8"]:
                Logger.logr.info("Starting Running Para2vec Baseline for Window = %s" %window)
                self.postgres_recorder.truncateSummaryTable()
                paraBaseline = P2VSENTCExecutableRunner(self.dbstring)
                if  window=="8":  
                    paraBaseline.prepareData(pd)
                paraBaseline.runTheBaseline(rbase,latent_space_size, window)
                val = paraBaseline.evaluateRankCorrelation(dataset)
                paraBaseline.doHouseKeeping()           
                spearman[window] = val
                Logger.logr.info("correlation for %s = %s" %(window, spearman[window]))
            window_opt = max(spearman, key=spearman.get) #get the window for the max recall
            f.write("Optimal window size is %s%s"%(window_opt, os.linesep))
            f.write("spearman: %s%s" %(spearman, os.linesep))
            f.flush()

           

            # n2vBaseline = Node2VecRunner(self.dbstring)
            # n2vBaseline.prepareData(pd)
            # generate_walk = True 
            # n2vBaseline.runTheBaseline(rbase, latent_space_size, generate_walk)
            # n2vBaseline.doHouseKeeping()

            # spearman ={}
            # joint_beta_opt = None 
            # lambda_list = [0.3, 0.5, 0.8, 1.0]
            
            # #lambda_list = [0.3]   
            # for lambda_ in  lambda_list:
            #     Logger.logr.info("Starting running jl with lambda = %s" %(lambda_))
            #     self.postgres_recorder.truncateSummaryTable()
            #     os.environ["NBR_TYPE"]=str(0)
            #     os.environ["FULL_DATA"]=str(1)
            #     os.environ["LAMBDA"]=str(lambda_)
            #     jointL = JointLearningSen2VecRunner(self.dbstring)
            #     jointL.window = window_opt
            #     if lambda_==lambda_list[0]:
            #         jointL.prepareData(pd)
            #     jointL.runTheBaseline(rbase, latent_space_size)
            #     val =jointL.evaluateRankCorrelation(dataset)
            #     jointL.doHouseKeeping()           
            #     spearman[lambda_] = val
            #     Logger.logr.info("correlation for %s = %s" %(lambda_, spearman[lambda_]))
            # joint_beta_opt_full_fixed = max(spearman, key=spearman.get) #get the window for the max recall
            # f.write("Optimal joint_beta_opt_full_fixed is %s%s"%(joint_beta_opt_full_fixed, os.linesep))
            # f.write("spearman: %s%s" %(spearman, os.linesep))
            # f.flush()


            # spearman = {}
            # joint_beta_opt = None 
            # lambda_list = [0.3, 0.5, 0.8, 1.0]
            # #lambda_list = [0.3]   
            # for lambda_ in  lambda_list:
            #     Logger.logr.info("Starting running jl with lambda = %s" %(lambda_))
            #     self.postgres_recorder.truncateSummaryTable()
            #     os.environ["NBR_TYPE"]=str(1)
            #     os.environ["FULL_DATA"]=str(1)
            #     os.environ["LAMBDA"]=str(lambda_)
            #     jointL = JointLearningSen2VecRunner(self.dbstring)
            #     jointL.window = window_opt
            #     if lambda_==lambda_list[0]:
            #         jointL.prepareData(pd)
            #     jointL.runTheBaseline(rbase, latent_space_size)
            #     val = jointL.evaluateRankCorrelation(dataset)
            #     jointL.doHouseKeeping()           
            #     spearman[lambda_] = val
            #     Logger.logr.info("correlation for %s = %s" %(lambda_, spearman[lambda_]))
            # joint_beta_opt_full_n2v = max(spearman, key=spearman.get) #get the window for the max recall
            # f.write("Optimal joint_beta_opt_full_n2v is %s%s"%(joint_beta_opt_full_n2v, os.linesep))
            # f.write("spearman: %s%s" %(spearman, os.linesep))
            # f.flush()

            # spearman = {}
            # joint_beta_opt = None 
            # lambda_list = [0.3, 0.5, 0.8, 1.0]
          
            # #lambda_list = [0.3]      
            # for lambda_ in  lambda_list:
            #     Logger.logr.info("Starting running jl with lambda = %s" %(lambda_))
            #     self.postgres_recorder.truncateSummaryTable()
            #     os.environ["NBR_TYPE"]=str(0)
            #     os.environ["FULL_DATA"]=str(0)
            #     os.environ["LAMBDA"]=str(lambda_)
            #     jointL = JointLearningSen2VecRunner(self.dbstring)
            #     jointL.window = window_opt
            #     if lambda_==lambda_list[0]:
            #         jointL.prepareData(pd)
            #     jointL.runTheBaseline(rbase, latent_space_size)
            #     val =jointL.evaluateRankCorrelation(dataset)
            #     jointL.doHouseKeeping()           
            #     spearman[lambda_] = val
            #     Logger.logr.info("correlation for %s = %s" %(lambda_, spearman[lambda_]))
            # joint_beta_opt_random_fixed = max(spearman, key=spearman.get) #get the window for the max recall
            # f.write("Optimal joint_beta_opt_random_fixed is %s%s"%(joint_beta_opt_random_fixed, os.linesep))
            # f.write("spearman: %s%s" %(spearman, os.linesep))
            # f.flush()

            # spearman = {}
            # joint_beta_opt = None 
            # lambda_list = [0.3, 0.5, 0.8, 1.0]
           
            # #lambda_list = [0.3]       
            # for lambda_ in  lambda_list:
            #     Logger.logr.info("Starting running jl with lambda = %s" %(lambda_))
            #     self.postgres_recorder.truncateSummaryTable()
            #     os.environ["NBR_TYPE"]=str(1)
            #     os.environ["FULL_DATA"]=str(0)
            #     os.environ["LAMBDA"]=str(lambda_)
            #     jointL = JointLearningSen2VecRunner(self.dbstring)
            #     jointL.window = window_opt
            #     if lambda_==lambda_list[0]:
            #         jointL.prepareData(pd)
            #     jointL.runTheBaseline(rbase, latent_space_size)
            #     val =jointL.evaluateRankCorrelation(dataset)
            #     jointL.doHouseKeeping()           
            #     spearman[lambda_] = val
            #     Logger.logr.info("correlation for %s = %s" %(lambda_, spearman[lambda_]))
            # joint_beta_opt_random_n2v = max(spearman, key=spearman.get) #get the window for the max recall
            # f.write("Optimal joint_beta_opt_random_n2v is %s%s"%(joint_beta_opt_random_n2v, os.linesep))
            # f.write("spearman: %s%s" %(spearman, os.linesep))
            # f.flush()
    
            
# ######## Test ########################################
            os.environ["EVAL"]='TEST'
            
            self.sp_methods = {}
            self.pearson_methods = {}
            niter = 5

           

            for i in range(0,niter):
                f.write("###### Iteration: %s ######%s" %(i, os.linesep))
                f.write("Optimal Window: %s%s" %(window_opt, os.linesep))           
                self.postgres_recorder.truncateSummaryTable()
                paraBaseline = P2VSENTCExecutableRunner(self.dbstring)
                paraBaseline.runTheBaseline(rbase,latent_space_size, window_opt) #we need the p2v vectors created with optimal window
                sp,pearson = paraBaseline.evaluateRankCorrelation(dataset)
                paraBaseline.doHouseKeeping()
                self.insertevals(sp, pearson, paraBaseline.latReprName)

                f.flush()


                iterrunner = IterativeUpdateRetrofitRunner(self.dbstring)
                iterrunner.myalpha = alpha #reinitializing myalpha
                iterrunner.prepareData(pd)
                iterrunner.runTheBaseline(rbase)    
                sp, pearson = iterrunner.evaluateRankCorrelation(dataset)
                iterrunner.doHouseKeeping()
                self.insertevals(sp, pearson, paraBaseline.latReprName)


                # os.environ["NBR_TYPE"]=str(0)
                # os.environ["FULL_DATA"]=str(1)
                # os.environ["LAMBDA"]=str(joint_beta_opt_full_fixed)
                # jointL = JointLearningSen2VecRunner(self.dbstring)
                # jointL.window = window_opt
                # jointL.prepareData(pd)
                # jointL.runTheBaseline(rbase, latent_space_size)
                # sp,pearson = jointL.evaluateRankCorrelation(dataset)
                # jointL.doHouseKeeping()
                # self.insertevals(sp, pearson, jointL.latReprName)

                # f.flush()

                # os.environ["NBR_TYPE"]=str(1)
                # os.environ["FULL_DATA"]=str(1)
                # os.environ["LAMBDA"]=str(joint_beta_opt_full_n2v)
                # jointL = JointLearningSen2VecRunner(self.dbstring)
                # jointL.window = window_opt
                # jointL.prepareData(pd)
                # jointL.runTheBaseline(rbase, latent_space_size)
                # sp,pearson = jointL.evaluateRankCorrelation(dataset)
                # jointL.doHouseKeeping()
                # self.insertevals(sp, pearson, jointL.latReprName)

                # f.flush()
               
                # os.environ["NBR_TYPE"]=str(0)
                # os.environ["FULL_DATA"]=str(0)
                # os.environ["LAMBDA"]=str(joint_beta_opt_random_fixed)
                # jointL = JointLearningSen2VecRunner(self.dbstring)
                # jointL.window = window_opt
                # jointL.prepareData(pd)
                # jointL.runTheBaseline(rbase, latent_space_size)
                # sp,pearson = jointL.evaluateRankCorrelation(dataset)
                # jointL.doHouseKeeping()
                # self.insertevals(sp, pearson, jointL.latReprName)
                # f.flush()
               
                # os.environ["NBR_TYPE"]=str(1)
                # os.environ["FULL_DATA"]=str(0)
                # os.environ["LAMBDA"]=str(joint_beta_opt_random_n2v)
                # jointL = JointLearningSen2VecRunner(self.dbstring)
                # jointL.window = window_opt
                # jointL.prepareData(pd)
                # jointL.runTheBaseline(rbase, latent_space_size)
                # sp,pearson = jointL.evaluateRankCorrelation(dataset)
                # jointL.doHouseKeeping()
                # self.insertevals(sp, pearson, jointL.latReprName)
                # f.flush()

        for k,v in sorted(self.sp_methods.items()):
            print (k, sum(v) / float(len(v)))

        print (os.linesep, os.linesep)
        for k,v in sorted(self.pearson_methods.items()):
            print (k, sum(v) / float(len(v)))
