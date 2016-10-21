#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os 
import sys 
import math
from abc import ABCMeta, abstractmethod
import networkx as nx 
import pandas as pd
from log_manager.log_config import Logger
from sklearn.dummy import DummyClassifier
from sklearn import linear_model
import operator
import numpy as np 
import subprocess 
from db_connector.PostgresPythonConnector import PostgresPythonConnector
from evaluation.classificationevaluaiton.ClassificationEvaluation import ClassificationEvaluation 
from evaluation.clusteringevaluation.ClusteringEvaluation import ClusteringEvaluation 

class BaselineRunner:
	def __init__(self, dbstring, **kwargs):
		"""
		"""
		self.postgresConnection = PostgresPythonConnector(dbstring)

	def _runProcess (self,args): 
		Logger.logr.info(args)
		proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		out, err = proc.communicate()
		if 	proc.returncode != 0: 
			Logger.logr.error("Process haven't terminated successfully")
			Logger.logr.info(out)
			Logger.logr.info(err)
			sys.exit(1)

	def _runClassificationValidation(self, summaryMethodID,  reprName, vDict):
		classeval = ClassificationEvaluation(postgres_connection=self.postgresConnection)
		classeval.generateDataValidation(summaryMethodID, reprName, vDict)
		classeval.runClassificationTask(summaryMethodID, reprName)

	def _runClusteringValidation(self, summaryMethodID, reprName, vDict):
		clusterEval = ClusteringEvaluation(postgres_connection=self.postgresConnection)
		clusterEval.generateDataValidation(summaryMethodID, reprName, vDict)
		clusterEval.runClusteringTask(summaryMethodID, reprName)

	def _runClassification(self, summaryMethodID,  reprName, vDict):
		classeval = ClassificationEvaluation(postgres_connection=self.postgresConnection)
		classeval.generateData(summaryMethodID, reprName, vDict)
		classeval.runClassificationTask(summaryMethodID, reprName)

	def _runClustering(self, summaryMethodID, reprName, vDict):
		clusterEval = ClusteringEvaluation(postgres_connection=self.postgresConnection)
		clusterEval.generateData(summaryMethodID, reprName, vDict)
		clusterEval.runClusteringTask(summaryMethodID, reprName)

	@abstractmethod
	def prepareData(self):
		"""
		"""
		pass

	@abstractmethod
	def runTheBaseline(self):
		"""
		"""
		pass

	@abstractmethod
	def runEvaluationTask(self):
		"""
		"""
		pass

	@abstractmethod
	def generateSummary(self):
		"""
		"""
		pass 
	@abstractmethod
	def doHouseKeeping(self):
		"""
		This method will close existing database connections and 
		other resouces it has used. 
		"""
		pass

	def __getAdjustedMutulScore(self, latreprName):
		file_ = os.path.join(os.environ["TRTESTFOLDER"], "%s_clustereval_2.txt"%latreprName)
		for line in open(file_):
			if "Adjusted Mutual Info Score:" in line:
				adj_score = line.strip()[line.strip().find("Adjusted Mutual Info Score:"):]
				adj_score = float(adj_score)
				return adj_score

	def __getF1(self, latreprName):
		"""
		"""
		file_ = os.path.join(os.environ["TRTESTFOLDER"], "%s_raweval_2.txt"%latreprName)
		for line in open(file_):
			if "avg" in line:
				line_elems = line.strip().split()
				f1 = float(line_elems[5])
				return f1 

	def __writeResult(self, latreprName, f):
		if os.environ['TEST_FOR'] == 'CLASS':
			file_ = os.path.join(os.environ["TRTESTFOLDER"], "%s_raweval_2.txt"%latreprName)
			for line in open(file_):
				f.write(line)
		else:
			file_ = os.path.join(os.environ["TRTESTFOLDER"], "%s_clustereval_2.txt"%latreprName)
			for line in open(file_):
				f.write(line)


	def _runClassificationValidation(self, pd, rbase, gs, dataset_name):
		############# Validation ############################		
		with open('%s%s%s%s' %(os.environ["TRTESTFOLDER"],"/",dataset_name,"_hyperparameters_class.txt"), 'w') as f:
			
			latent_space_size = 300
			os.environ['EVAL'] = 'VALID'
			os.environ['VALID_FOR'] = 'CLASS'
	
			f1 = {}
			window_opt = None #var for the optimal window
			#for window in ["8", "10", "12"]:
			for window in ["8"]:
				Logger.logr.info("Starting Running Para2vec Baseline for Window = %s" %window)
				self.postgres_recorder.truncateSummaryTable()
				paraBaseline = P2VSENTCExecutableRunner(self.dbstring)
				if 	window=="8":  
					paraBaseline.prepareData(pd)
				paraBaseline.runTheBaseline(rbase,latent_space_size, window)
				paraBaseline.generateSummary(gs)		
				paraBaseline.runEvaluationTask()
				paraBaseline.doHouseKeeping()	
				f1[window] = self.__getF1(paraBaseline.latReprName)
				Logger.logr.info("F1 for %s = %s" %(window, f1[window]))
			window_opt = max(f1, key=f1.get) 
			f.write("Optimal window size is %s%s"%(window_opt, os.linesep))
			f.write("P2V Window F1s: %s%s" %(f1, os.linesep))
			f.flush()

			Logger.logr.info("Starting Running Para2vec Baseline for Optimal Window = %s" %window_opt)
			self.postgres_recorder.truncateSummaryTable()
			paraBaseline = P2VSENTCExecutableRunner(self.dbstring)
			paraBaseline.runTheBaseline(rbase,latent_space_size, window_opt)
			paraBaseline.generateSummary(gs)
			#we need the p2v vectors created with optimal window
			paraBaseline.doHouseKeeping()

			f1 = {}
			beta_opt = None #var for the optimal beta
			#for beta in ["0.3", "0.6", "0.9","1.0"]:
			for beta in ["0.3"]:
				Logger.logr.info("Starting Running Node2vec Baseline for Beta = %s" %beta)
				n2vBaseline = Node2VecRunner(self.dbstring)
				n2vBaseline.mybeta = beta #reinitializing mybeta
				generate_walk = False
				if beta=="0.3":
				   n2vBaseline.prepareData(pd)
				   generate_walk = True 
				n2vBaseline.runTheBaseline(rbase, latent_space_size, generate_walk)
				n2vBaseline.runEvaluationTask()
				n2vBaseline.doHouseKeeping()
				f1[beta_opt] = self.__getF1("%s_retrofit"%n2vBaseline.latReprName)
				Logger.logr.info("F1 for %s = %s" %(beta, f1[window]))
			beta_opt = max(f1, key=f1.get)  
			f.write("Optimal Beta is %s%s"%(beta_opt, os.linesep))
			f.write("N2V Beta f1s: %s%s" %(f1, os.linesep))
			f.flush()
			
			f1 = {}
			alpha_opt = None #var for the optimal beta
			#for alpha in [0.3, 0.6, 0.8, 1.0]:
			for alpha in [0.3]:
				Logger.logr.info("Starting Running Iterative Baseline for Alpha = %s" %alpha)
				iterrunner = IterativeUpdateRetrofitRunner(self.dbstring)
				iterrunner.myalpha = alpha #reinitializing myalpha
				if alpha==0.3:
					iterrunner.prepareData(pd)
				iterrunner.runTheBaseline(rbase)	
				iterrunner.runEvaluationTask()
				iterrunner.doHouseKeeping()
				f1[beta_opt] = self.__getF1("%s_weighted"%iterrunner.latReprName)	
				Logger.logr.info("F1 for %s = %s" %(alpha, f1[alpha]))
			alpha_opt = max(f1, key=f1.get) #get the alpha for the max recall
			Logger.logr.info("Optimal Alpha=%s" %alpha_opt)
			f.write("Optimal alpha is %.2f%s"%(alpha_opt, os.linesep))
			f.write("ITR Alpha f1s: %s%s" %(recalls, os.linesep))
			f.flush()

			w_f1 = {}
			unw_f1 = {}
			w_opt = None
			unw_opt = None
			#for beta in [0.3, 0.6, 0.8, 1.0]:
			for beta in [0.3]:
				Logger.logr.info("Starting Running Regularized Baseline for Beta = %s" %beta)
				self.postgres_recorder.truncateSummaryTable()
				regs2v = RegularizedSen2VecRunner(self.dbstring)
				regs2v.regBetaW = beta
				regs2v.regBetaUNW = beta
				if beta==0.3:
					regs2v.prepareData(pd)
				regs2v.runTheBaseline(rbase, latent_space_size)
				regs2v.runEvaluationTask()
				regs2v.doHouseKeeping()
				w_f1[beta] = self.__getF1("%s_neighbor_w"%regs2v.latReprName)	
				unw_f1[beta] = self.__getF1("%s_neighbor_unw"%regs2v.latReprName)	
				Logger.logr.info("W_f1 for %s = %s" %(beta, w_f1[beta]))
				Logger.logr.info("UNW_f1 for %s = %s" %(beta, unw_f1[beta]))
			w_opt_reg = max(w_f1, key=w_recalls.get)
			unw_opt_reg = max(unw_f1, key=unw_recalls.get)
			Logger.logr.info("Optimal regBetaW=%s and regBetaUNW=%s" %(w_opt_reg, unw_opt_reg))

			f.write("Optimal REG BetaW : %.2f%s" %(w_opt_reg, os.linesep))
			f.write("Optimal REG BetaUNW : %.2f%s" %(unw_opt_reg, os.linesep))
			f.write("REG BetaW f1s: %s%s" %(w_recalls, os.linesep))
			f.write("REG BetaUNW f1s: %s%s" %(unw_recalls, os.linesep))
			f.flush()

			w_f1 = {}
			unw_f1 = {}
			w_opt = None
			unw_opt = None
			#for beta in [0.3, 0.6, 0.8, 1.0]:
			for beta in [0.3]:
				Logger.logr.info("Starting Running Dict Regularized Baseline for Beta = %s" %beta)
				dictregs2v = DictRegularizedSen2VecRunner(self.dbstring)
				dictregs2v.dictregBetaW = beta
				dictregs2v.dictregBetaUNW = beta
				if beta==0.3:
					dictregs2v.prepareData(pd)
				dictregs2v.runTheBaseline(rbase, latent_space_size)
				dictregs2v.runEvaluationTask()
				dictregs2v.doHouseKeeping()
				w_recalls[beta] = self.__getF1("%s_neighbor_w"%dictregs2v.latReprName)	
				unw_recalls[beta] = self.__getF1("%s_neighbor_unw"%dictregs2v.latReprName)	
				Logger.logr.info("W_f1 for %s = %s" %(beta, w_f1[beta]))
				Logger.logr.info("UNW_f1 for %s = %s" %(beta, unw_f1[beta]))
			w_opt_dict_reg = max(w_f1, key=w_f1.get)
			unw_opt_dict_reg = max(unw_f1, key=unw_f1.get)
			Logger.logr.info("Optimal dictregBetaW=%s and dictregBetaUNW=%s" %(w_opt_dict_reg, unw_opt_dict_reg))

			f.write("DCT BetaW: %.2f%s" %(w_opt_dict_reg, os.linesep))
			f.write("DCT BetaUNW: %.2f%s" %(unw_opt_dict_reg, os.linesep))
			f.write("DCT BetaW f1s: %s%s" %(w_f1, os.linesep))
			f.write("DCT BetaUNW f1s: %s%s" %(unw_f1, os.linesep))
			f.flush()

	def _runClusteringValidation(self, pd, rbase, gs, dataset_name):
		############# Validation ############################		
		with open('%s%s%s%s' %(os.environ["TRTESTFOLDER"],"/",dataset_name,"_hyperparameters_clust.txt"), 'w') as f:
			
			latent_space_size = 300
			os.environ['EVAL'] = 'VALID'
			os.environ['VALID_FOR'] = 'CLUST'
	
			adjustedMScore = {}
			window_opt = None #var for the optimal window
			#for window in ["8", "10", "12"]:
			for window in ["8"]:
				Logger.logr.info("Starting Running Para2vec Baseline for Window = %s" %window)
				self.postgres_recorder.truncateSummaryTable()
				paraBaseline = P2VSENTCExecutableRunner(self.dbstring)
				if 	window=="8":  
					paraBaseline.prepareData(pd)
				paraBaseline.runTheBaseline(rbase,latent_space_size, window)
				paraBaseline.generateSummary(gs)		
				paraBaseline.runEvaluationTask()
				paraBaseline.doHouseKeeping()	
				adjustedMScore[window] = self.__getAdjustedMutulScore(paraBaseline.latReprName)
				Logger.logr.info("adjustedMScore for %s = %s" %(window, adjustedMScore[window]))
			window_opt = max(adjustedMScore, key=adjustedMScore.get) 
			f.write("Optimal window size is %s%s"%(window_opt, os.linesep))
			f.write("P2V Window adjustedMScores: %s%s" %(adjustedMScore, os.linesep))
			f.flush()

			Logger.logr.info("Starting Running Para2vec Baseline for Optimal Window = %s" %window_opt)
			self.postgres_recorder.truncateSummaryTable()
			paraBaseline = P2VSENTCExecutableRunner(self.dbstring)
			paraBaseline.runTheBaseline(rbase,latent_space_size, window_opt)
			paraBaseline.generateSummary(gs)
			paraBaseline.doHouseKeeping()

			adjustedMScore = {}
			beta_opt = None #var for the optimal beta
			#for beta in ["0.3", "0.6", "0.9","1.0"]:
			for beta in ["0.3"]:
				Logger.logr.info("Starting Running Node2vec Baseline for Beta = %s" %beta)
				n2vBaseline = Node2VecRunner(self.dbstring)
				n2vBaseline.mybeta = beta #reinitializing mybeta
				generate_walk = False
				if beta=="0.3":
				   n2vBaseline.prepareData(pd)
				   generate_walk = True 
				n2vBaseline.runTheBaseline(rbase, latent_space_size, generate_walk)
				n2vBaseline.runEvaluationTask()
				n2vBaseline.doHouseKeeping()
				adjustedMScore[beta_opt] = self.__getAdjustedMutulScore("%s_retrofit"%n2vBaseline.latReprName)
				Logger.logr.info("Adjusted MScore for %s = %s" %(beta, adjustedMScore[window]))
			beta_opt = max(adjustedMScore, key=adjustedMScore.get)  
			f.write("Optimal Beta is %s%s"%(beta_opt, os.linesep))
			f.write("N2V Beta adjusted mutual score: %s%s" %(adjustedMScore, os.linesep))
			f.flush()
			
			adjustedMScore = {}
			alpha_opt = None #var for the optimal beta
			#for alpha in [0.3, 0.6, 0.8, 1.0]:
			for alpha in [0.3]:
				Logger.logr.info("Starting Running Iterative Baseline for Alpha = %s" %alpha)
				iterrunner = IterativeUpdateRetrofitRunner(self.dbstring)
				iterrunner.myalpha = alpha #reinitializing myalpha
				if 	alpha==0.3:
					iterrunner.prepareData(pd)
				iterrunner.runTheBaseline(rbase)	
				iterrunner.runEvaluationTask()
				iterrunner.doHouseKeeping()
				adjustedMScore[beta_opt] = self.__getAdjustedMutulScore("%s_weighted"%iterrunner.latReprName)	
				Logger.logr.info("Adjusted Mutual Score for %s = %s" %(alpha, adjustedMScore[alpha]))
			alpha_opt = max(adjustedMScore, key=adjustedMScore.get) #get the alpha for the max recall
			Logger.logr.info("Optimal Alpha=%s" %alpha_opt)
			f.write("Optimal alpha is %.2f%s"%(alpha_opt, os.linesep))
			f.write("ITR Alpha adjusted mutual scores: %s%s" %(adjustedMScore, os.linesep))
			f.flush()

			w_adjusted = {}
			unw_adjusted = {}
			w_opt = None
			unw_opt = None
			#for beta in [0.3, 0.6, 0.8, 1.0]:
			for beta in [0.3]:
				Logger.logr.info("Starting Running Regularized Baseline for Beta = %s" %beta)
				self.postgres_recorder.truncateSummaryTable()
				regs2v = RegularizedSen2VecRunner(self.dbstring)
				regs2v.regBetaW = beta
				regs2v.regBetaUNW = beta
				if beta==0.3:
					regs2v.prepareData(pd)
				regs2v.runTheBaseline(rbase, latent_space_size)
				regs2v.runEvaluationTask()
				regs2v.doHouseKeeping()
				w_adjusted[beta] = self.__getAdjustedMutulScore("%s_neighbor_w"%regs2v.latReprName)	
				unw_adjusted[beta] = self.__getAdjustedMutulScore("%s_neighbor_unw"%regs2v.latReprName)	
				Logger.logr.info("W_adjusted for %s = %s" %(beta, w_adjusted[beta]))
				Logger.logr.info("UNW_adjusted for %s = %s" %(beta, unw_adjusted[beta]))
			w_opt_reg = max(w_adjusted, key=w_recalls.get)
			unw_opt_reg = max(unw_adjusted, key=unw_recalls.get)
			Logger.logr.info("Optimal regBetaW=%s and regBetaUNW=%s" %(w_opt_reg, unw_opt_reg))

			f.write("Optimal REG BetaW : %.2f%s" %(w_opt_reg, os.linesep))
			f.write("Optimal REG BetaUNW : %.2f%s" %(unw_opt_reg, os.linesep))
			f.write("REG BetaW adjusted mutual score: %s%s" %(w_adjusted, os.linesep))
			f.write("REG BetaUNW adjusted mutual score: %s%s" %(unw_adjusted, os.linesep))
			f.flush()

			w_adjusted = {}
			unw_adjusted = {}
			w_opt = None
			unw_opt = None
			#for beta in [0.3, 0.6, 0.8, 1.0]:
			for beta in [0.3]:
				Logger.logr.info("Starting Running Dict Regularized Baseline for Beta = %s" %beta)
				dictregs2v = DictRegularizedSen2VecRunner(self.dbstring)
				dictregs2v.dictregBetaW = beta
				dictregs2v.dictregBetaUNW = beta
				if beta==0.3:
					dictregs2v.prepareData(pd)
				dictregs2v.runTheBaseline(rbase, latent_space_size)
				dictregs2v.runEvaluationTask()
				dictregs2v.doHouseKeeping()
				w_adjusted[beta] = self.__getAdjustedMutulScore("%s_neighbor_w"%dictregs2v.latReprName)	
				unw_adjusted[beta] = self.__getAdjustedMutulScore("%s_neighbor_unw"%dictregs2v.latReprName)	
				Logger.logr.info("W_adjusted for %s = %s" %(beta, w_adjusted[beta]))
				Logger.logr.info("UNW_adjusted for %s = %s" %(beta, unw_adjusted[beta]))
			w_opt_dict_reg = max(w_adjusted, key=w_adjusted.get)
			unw_opt_dict_reg = max(unw_adjusted, key=unw_adjusted.get)
			Logger.logr.info("Optimal dictregBetaW=%s and dictregBetaUNW=%s" %(w_opt_dict_reg, unw_opt_dict_reg))

			f.write("DCT BetaW: %.2f%s" %(w_opt_dict_reg, os.linesep))
			f.write("DCT BetaUNW: %.2f%s" %(unw_opt_dict_reg, os.linesep))
			f.write("DCT BetaW adjusted mutual score: %s%s" %(w_adjusted, os.linesep))
			f.write("DCT BetaUNW adjusted mutual score: %s%s" %(unw_adjusted, os.linesep))
			f.flush()


	def doTesting(self, optPDict, dataset_name, classification=True):
		######### Test ########################################
		os.environ["EVAL"]='TEST'

		if classification==True:
			os.environ['TEST_FOR'] = 'CLASS'
		else:
			os.environ['TEST_FOR'] = 'CLUST'

		with open('%s%s%s%s' %(os.environ["TRTESTFOLDER"],"/",dataset_name,"_testresults_%s.txt"%os.environ['VALID_FOR']), 'w') as f:
		niter = 5
		for i in range(0,niter):
			f.write("###### Iteration: %s ######%s" %(i, os.linesep))
			f.write("Optimal Window: %s%s" %(window_opt, os.linesep))				
			self.postgres_recorder.truncateSummaryTable()
			paraBaseline = P2VSENTCExecutableRunner(self.dbstring)
			paraBaseline.runTheBaseline(rbase,latent_space_size, window_opt) 
			paraBaseline.generateSummary(gs)
			paraBaseline.runEvaluationTask()
			self.__writeResult(paraBaseline.latReprName, f)
			paraBaseline.doHouseKeeping()
			f.flush()


			f.write("Optimal Beta: %s%s" %(beta_opt, os.linesep))	
			n2vBaseline = Node2VecRunner(self.dbstring)
			n2vBaseline.mybeta = beta_opt
			generate_walk = False
			n2vBaseline.runTheBaseline(rbase, latent_space_size, generate_walk)
			n2vBaseline.runEvaluationTask()
			self.__writeResult("%s"%n2vBaseline.latreprName, f)
			self.__writeResult("%s_init"%n2vBaseline.latreprName, f)
			self.__writeResult("%s_retrofit"%n2vBaseline.latreprName, f)
			n2vBaseline.doHouseKeeping()
			f.flush()


			f.write("Optimal alpha: %.2f%s" %(alpha_opt, os.linesep))	
			iterrunner = IterativeUpdateRetrofitRunner(self.dbstring)
			iterrunner.myalpha = alpha_opt #reinitializing myalpha
			iterrunner.runTheBaseline(rbase)
			iterrunner.runEvaluationTask()
			self.__writeResult("%s_unweighted"%iterrunner.latreprName, f)
			self.__writeResult("%s_weighted"%iterrunner.latreprName, f)
			iterrunner.doHouseKeeping()


			f.write("Optimal regBetaW: %.2f%s" %(w_opt_reg, os.linesep))	
			f.write("Optimal regBetaUNW: %.2f%s" %(unw_opt_reg, os.linesep))	
			regs2v = RegularizedSen2VecRunner(self.dbstring)
			regs2v.regBetaW = w_opt_reg 
			regs2v.regBetaUNW = unw_opt_reg
			regs2v.runTheBaseline(rbase, latent_space_size)
			self.__writeResult("%s_neighbor_w"%regs2v.latreprName, f)
			self.__writeResult("%s_neighbor_unw"%regs2v.latreprName, f)
			regs2v.runEvaluationTask()
			regs2v.doHouseKeeping()
			f.flush()


			f.write("Optimal regBetaW: %.2f%s" %(w_opt_dict_reg, os.linesep))	
			f.write("Optimal regBetaUNW: %.2f%s" %(unw_opt_dict_reg, os.linesep))	
			dictregs2v = DictRegularizedSen2VecRunner(self.dbstring)
			dictregs2v.dictregBetaW = w_opt_dict_reg
			dictregs2v.dictregBetaUNW = unw_opt_dict_reg
			dictregs2v.runTheBaseline(rbase, latent_space_size)
			dictregs2v.runEvaluationTask()
			self.__writeResult("%s_neighbor_w"%regs2v.latreprName, f)
			self.__writeResult("%s_neighbor_unw"%regs2v.latreprName, f)
			dictregs2v.doHouseKeeping()
			f.flush()