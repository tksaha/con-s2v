#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os 
import re
import numpy as np 
import pickle
import operator
import logging 
from documentReader.DocumentReader import DocumentReader
from documentReader.PostgresDataRecorder   import PostgresDataRecorder
from log_manager.log_config import Logger


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
        os.environ['TEST_AND_TRAIN'] = 'NO'
    
        
        
        latent_space_size = 300

        ############# Validation ############################       
        with open('%s%s%s%s' %(os.environ["TRTESTFOLDER"],"/","sick","_hyperparameters.txt"), 'w') as f:
            os.environ['EVAL']='VALID'
    
            spearman = {}  #spearman or pearson correlation
         


            # lambda_list = [0.3, 0.5, 0.8, 1.0]
            # unw_corr = {}
            # for beta in lambda_list:
            # #for beta in [0.3]:
            #     Logger.logr.info("Starting Running Regularized Baseline for Beta = %s" %beta)
            #     regs2v = RegularizedSen2VecRunner(self.dbstring)
            #     regs2v.regBetaUNW = beta
            #     if beta==0.3:
            #        regs2v.prepareData(pd)
            #     regs2v.runTheBaseline(rbase, latent_space_size)
            #     val = regs2v.evaluateRankCorrelation(dataset)
            #     unw_corr[beta] = val 
            #     Logger.logr.info("UNW_corr for %s = %s" %(beta, unw_corr[beta]))
           
            # unw_opt_reg = max(unw_corr, key=unw_corr.get)
            # Logger.logr.info("Optimal regBetaUNW=%s" %(unw_opt_reg))
          


    
            
# ######## Test ########################################
            os.environ["EVAL"]='TEST'
            
            self.sp_methods = {}
            self.pearson_methods = {}
            niter = 5

           

            for i in range(0,niter):
                f.write("###### Iteration: %s ######%s" %(i, os.linesep))
               	Logger.logr.info("###### Iteration: %s ######%s" %(i, os.linesep))
 
	
                # regs2v = RegularizedSen2VecRunner(self.dbstring)
                # regs2v.regBetaUNW = unw_opt_reg
                # regs2v.prepareData(pd)
                # regs2v.runTheBaseline(rbase, latent_space_size)
                # sp, pearson = regs2v.evaluateRankCorrelation(dataset)
                # regs2v.doHouseKeeping()
                # self.insertevals(sp, pearson, regs2v.latReprName)
                # f.flush()
            

        for k,v in sorted(self.sp_methods.items()):
            print (k, sum(v) / float(len(v)))

        print (os.linesep, os.linesep)
        for k,v in sorted(self.pearson_methods.items()):
            print (k, sum(v) / float(len(v)))
