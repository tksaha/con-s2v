#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os 
import sys 
from abc import ABCMeta, abstractmethod
from log_manager.log_config import Logger 

import nltk
from nltk.tokenize import sent_tokenize
from utility.Utility import Utility
import gensim
from baselineRunner.Paragraph2VecSentenceRunner  import Paragraph2VecSentenceRunner 
from baselineRunner.Node2VecRunner import Node2VecRunner
from baselineRunner.IterativeUpdateRetrofitRunner import IterativeUpdateRetrofitRunner
from baselineRunner.P2VSENTCExecutableRunner import P2VSENTCExecutableRunner
from baselineRunner.RegularizedSen2VecRunner import RegularizedSen2VecRunner
from baselineRunner.DictRegularizedSen2VecRunner import DictRegularizedSen2VecRunner
from baselineRunner.JointLearningSen2VecRunner import JointLearningSen2VecRunner
from baselineRunner.FastSentVariantRunner import FastSentVariantRunner
from baselineRunner.JointSupervisedRunner import JointSupervisedRunner


class DocumentReader:
	"""
	DocumentReader Base
	"""
	__metaclass__ = ABCMeta

	def __init__(self):
		self.utFunction = Utility("Text Utility")


	@abstractmethod
	def readDocument(self):
		pass


	"""
	Protected Methods: Accessed by subclasses 
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

	
	def __getAdjustedMutulScore(self, latreprName):
		file_ = os.path.join(os.environ["TRTESTFOLDER"], "%s_rawclustereval_2.txt"%latreprName)
		for line in open(file_):
			if "Adjusted Mutual Info Score:" in line:
				adj_score = line.strip()[line.strip().rfind(":")+1:]
				adj_score = float(adj_score)
				Logger.logr.info("Returning value %.2f"%adj_score)
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
			file_ = os.path.join(os.environ["TRTESTFOLDER"], "%s_rawclustereval_2.txt"%latreprName)
			for line in open(file_):
				f.write(line)


	def _runClassificationOnValidation(self, pd, rbase, gs, dataset_name):
		############# Validation ############################	
		optPDict = {}	
		with open('%s%s%s%s' %(os.environ["TRTESTFOLDER"],"/",dataset_name,"_hyperparameters_class.txt"), 'w') as f:
			
			latent_space_size = 300
			os.environ['EVAL'] = 'VALID'
			os.environ['VALID_FOR'] = 'CLASS'

			generate_walk = True 
	
			# f1 = {}
			# window_opt = None #var for the optimal window
			# window_size_list = ["8", "10", "12"]
			# for window in window_size_list:
			# 	Logger.logr.info("Starting Running Para2vec Baseline for Window = %s" %window)				
			# 	paraBaseline = P2VSENTCExecutableRunner(self.dbstring)
			# 	if 	window == window_size_list[0]:  
			# 		self.postgres_recorder.truncateSummaryTable()
			# 		paraBaseline.prepareData(pd)		
			# 	paraBaseline.runTheBaseline(rbase,latent_space_size, window)
			# 	if window == window_size_list[0]:
			# 		paraBaseline.generateSummary(gs)
			# 	paraBaseline.runEvaluationTask()
			# 	paraBaseline.doHouseKeeping()	
			# 	f1[window] = self.__getF1(paraBaseline.latReprName)
			# 	Logger.logr.info("F1 for %s = %s" %(window, f1[window]))
			# window_opt = max(f1, key=f1.get) 
			# f.write("Optimal window size is %s%s"%(window_opt, os.linesep))
			# f.write("P2V Window F1s: %s%s" %(f1, os.linesep))
			# f.flush()
			# optPDict["window"] = window_opt

			# Logger.logr.info("Starting Running Para2vec Baseline for Optimal Window = %s" %window_opt)
			
			# paraBaseline = P2VSENTCExecutableRunner(self.dbstring)
			# paraBaseline.runTheBaseline(rbase,latent_space_size, window_opt)
			# self.postgres_recorder.truncateSummaryTable()
			# paraBaseline.generateSummary(gs)
			# paraBaseline.doHouseKeeping()

			# if generate_walk == True:
			#    n2vBaseline = Node2VecRunner(self.dbstring)
			#    n2vBaseline.prepareData(pd)
			#    n2vBaseline.runTheBaseline(rbase, latent_space_size, generate_walk)
			#    generate_walk = False 
			
			# for full fixed nbr
			f1 = {}
			optPDict["window"] = "10"
			joint_beta_opt = None #var for the optimal joint_beta
	
			lambda_list = [0.3, 0.5, 0.8, 1.0]
			#lambda_list = [0.3]		
			for lambda_ in  lambda_list:
				Logger.logr.info("Starting running jl with lambda = %s" %(lambda_))
				os.environ["NBR_TYPE"]=str(0)
				os.environ["FULL_DATA"]=str(1)
				os.environ["LAMBDA"]=str(lambda_)
				jointL = JointLearningSen2VecRunner(self.dbstring)
				jointL.window = optPDict["window"]
				if lambda_==lambda_list[0]:
			   		jointL.prepareData(pd)
				jointL.runTheBaseline(rbase, latent_space_size)
				jointL.runEvaluationTask()
				jointL.doHouseKeeping()
				f1[lambda_] = self.__getF1("%s" %jointL.latReprName)
				Logger.logr.info("F1 for lambda,%s = %s" %(lambda_,f1[lambda_]))
			joint_beta_opt = max(f1, key=f1.get) 
	
			Logger.logr.info("Optimal lambda for full fixed = %s" %joint_beta_opt)		
			f.write("Optimal lambda for full fixed nbr is %.2f%s"%(joint_beta_opt, os.linesep))
			f.write("JTL Joint-Beta f1s: %s%s" %(f1, os.linesep))
			f.flush()
			optPDict['lambda-full-fixed'] = joint_beta_opt

			f1 = {}
			joint_beta_opt = None
			lambda_list = [0.3, 0.5, 0.8, 1.0]
			#lambda_list = [0.3]		
			for lambda_ in  lambda_list:
				Logger.logr.info("Starting running jl with lambda = %s" %(lambda_))
				os.environ["NBR_TYPE"]=str(1)
				os.environ["FULL_DATA"]=str(1)
				os.environ["LAMBDA"]=str(lambda_)
				jointL = JointLearningSen2VecRunner(self.dbstring)
				jointL.window = optPDict["window"]
				if lambda_==lambda_list[0]:
			   		jointL.prepareData(pd)
				jointL.runTheBaseline(rbase, latent_space_size)
				jointL.runEvaluationTask()
				jointL.doHouseKeeping()
				f1[lambda_] = self.__getF1("%s" %jointL.latReprName)
				Logger.logr.info("F1 for lambda,%s = %s" %(lambda_,f1[lambda_]))
			joint_beta_opt = max(f1, key=f1.get) 
	
			Logger.logr.info("Optimal lambda for full n2v = %s" %joint_beta_opt)		
			f.write("Optimal lambda for full n2v nbr is %.2f%s"%(joint_beta_opt, os.linesep))
			f.write("JTL Joint-Beta f1s: %s%s" %(f1, os.linesep))
			f.flush()
			optPDict['lambda-full-n2v'] = joint_beta_opt


			f1 = {}
			joint_beta_opt = None
			lambda_list = [0.3, 0.5, 0.8, 1.0]
			#lambda_list = [0.3]		
			for lambda_ in  lambda_list:
				Logger.logr.info("Starting running jl with lambda = %s" %(lambda_))
				os.environ["NBR_TYPE"]=str(0)
				os.environ["FULL_DATA"]=str(0)
				os.environ["LAMBDA"]=str(lambda_)
				jointL = JointLearningSen2VecRunner(self.dbstring)
				jointL.window = optPDict["window"]
				if lambda_==lambda_list[0]:
			   		jointL.prepareData(pd)
				jointL.runTheBaseline(rbase, latent_space_size)
				jointL.runEvaluationTask()
				jointL.doHouseKeeping()
				f1[lambda_] = self.__getF1("%s" %jointL.latReprName)
				Logger.logr.info("F1 for lambda,%s = %s" %(lambda_,f1[lambda_]))
			joint_beta_opt = max(f1, key=f1.get) 
	
			Logger.logr.info("Optimal lambda for random fixed = %s" %joint_beta_opt)		
			f.write("Optimal lambda for random fixed nbr is %.2f%s"%(joint_beta_opt, os.linesep))
			f.write("JTL Joint-Beta f1s: %s%s" %(f1, os.linesep))
			f.flush()
			optPDict['lambda-random-fixed'] = joint_beta_opt

			f1 = {}
			joint_beta_opt = None
			#lambda_list = [0.3, 0.5, 0.8, 1.0]
				
			for lambda_ in  lambda_list:
				Logger.logr.info("Starting running jl with lambda = %s" %(lambda_))
				os.environ["NBR_TYPE"]=str(1)
				os.environ["FULL_DATA"]=str(0)
				os.environ["LAMBDA"]=str(lambda_)
				jointL = JointLearningSen2VecRunner(self.dbstring)
				jointL.window = optPDict["window"]
				if lambda_==lambda_list[0]:
			   		jointL.prepareData(pd)
				jointL.runTheBaseline(rbase, latent_space_size)
				jointL.runEvaluationTask()
				jointL.doHouseKeeping()
				f1[lambda_] = self.__getF1("%s" %jointL.latReprName)
				Logger.logr.info("F1 for lambda,%s = %s" %(lambda_,f1[lambda_]))
			joint_beta_opt = max(f1, key=f1.get) 
	
			Logger.logr.info("Optimal lambda for random n2v = %s" %joint_beta_opt)		
			f.write("Optimal lambda for random n2v nbr is %.2f%s"%(joint_beta_opt, os.linesep))
			f.write("JTL Joint-Beta f1s: %s%s" %(f1, os.linesep))
			f.flush()
			optPDict['lambda-random-n2v'] = joint_beta_opt

		
		return optPDict


	def _runClusteringOnValidation(self, pd, rbase, gs, dataset_name):
		############# Validation ############################		
		optPDict = {}
		# with open('%s%s%s%s' %(os.environ["TRTESTFOLDER"],"/",dataset_name,"_hyperparameters_clust.txt"), 'w') as f:
			
		# 	latent_space_size = 300
		# 	os.environ['EVAL'] = 'VALID'
		# 	os.environ['VALID_FOR'] = 'CLUST'	

		# 	adjustedMScore = {}
		# 	window_opt = None #var for the optimal window
		# 	window_size_list = ["8", "10", "12"]
		# 	for window in window_size_list:
		# 	#for window in ["8"]:
		# 		Logger.logr.info("Starting Running Para2vec Baseline for Window = %s" %window)			
		# 		paraBaseline = P2VSENTCExecutableRunner(self.dbstring)
		# 		if 	window == window_size_list[0]: 
		# 			self.postgres_recorder.truncateSummaryTable()
		# 			paraBaseline.prepareData(pd)
					
		# 		paraBaseline.runTheBaseline(rbase,latent_space_size, window)
		# 		if window == window_size_list[0]:
		# 			paraBaseline.generateSummary(gs)
		# 		paraBaseline.runEvaluationTask()
		# 		paraBaseline.doHouseKeeping()	
		# 		adjustedMScore[window] = self.__getAdjustedMutulScore(paraBaseline.latReprName)
		# 		Logger.logr.info("adjustedMScore for %s = %s" %(window, adjustedMScore[window]))
		# 	window_opt = max(adjustedMScore, key=adjustedMScore.get) 
		# 	f.write("Optimal window size is %s%s"%(window_opt, os.linesep))
		# 	f.write("P2V Window adjustedMScores: %s%s" %(adjustedMScore, os.linesep))
		# 	f.flush()
		# 	optPDict["window"] = window_opt

		# 	Logger.logr.info("Starting Running Para2vec Baseline for Optimal Window = %s" %window_opt)
		# 	self.postgres_recorder.truncateSummaryTable()
		# 	paraBaseline = P2VSENTCExecutableRunner(self.dbstring)
		# 	paraBaseline.runTheBaseline(rbase,latent_space_size, window_opt)
		# 	paraBaseline.generateSummary(gs)
		# 	paraBaseline.doHouseKeeping()

			
			
		# 	adjustedMScore = {}
		# 	joint_beta_opt = None #var for the optimal beta
		# 	jointbeta_list = [0.5, 0.6, 0.7, 0.8, 0.85, 0.90, 0.95]
		# 	for joint_beta in jointbeta_list:
		# 		Logger.logr.info("Starting Running JointLearning Baseline for Joint-Beta = %s" %joint_beta)
		# 		jointL = JointLearningSen2VecRunner(self.dbstring)
		# 		jointL.jointbeta = joint_beta #reinitializing myalpha
		# 		jointL.window = optPDict["window"]
		# 		if 	joint_beta== jointbeta_list[0]:
		# 			jointL.prepareData(pd)
		# 		jointL.runTheBaseline(rbase, latent_space_size)	
		# 		jointL.runEvaluationTask()
		# 		jointL.doHouseKeeping()
		# 		adjustedMScore[joint_beta] = self.__getAdjustedMutulScore("%s"%jointL.latReprName)	
		# 		Logger.logr.info("Adjusted Mutual Score for %s = %s" %(joint_beta, adjustedMScore[joint_beta]))
		# 	joint_beta_opt = max(adjustedMScore, key=adjustedMScore.get) 
		# 	Logger.logr.info("Optimal Joint-Beta=%s" %joint_beta_opt)
		# 	f.write("Optimal Joint-Beta is %.2f%s"%(joint_beta_opt, os.linesep))
		# 	f.write("JTL Joint-Beta adjusted mutual scores: %s%s" %(adjustedMScore, os.linesep))
		# 	f.flush()
		# 	optPDict["joint-beta"] = joint_beta_opt

			

		return optPDict


	def doTesting(self, optPDict, dataset_name, rbase, pd, gs, classification=True):
		######### Test ########################################
		os.environ["EVAL"]='TEST'
		latent_space_size = 300

		if classification==True:
			os.environ['TEST_FOR'] = 'CLASS'
		else:
			os.environ['TEST_FOR'] = 'CLUST'

		f = open('%s%s%s%s' %(os.environ["TRTESTFOLDER"],"/",dataset_name,"_testresults_%s.txt"%os.environ['TEST_FOR']), 'w') 
		niter = 5
		for i in range(0,niter):
			#f.write("###### Iteration: %s ######%s" %(i, os.linesep))
			#f.write("Optimal Window: %s%s" %(optPDict["window"], os.linesep))				

			paraBaseline = P2VSENTCExecutableRunner(self.dbstring)
			paraBaseline.prepareData(pd)
			paraBaseline.runTheBaseline(rbase,latent_space_size, optPDict["window"])
			paraBaseline.runEvaluationTask()
			self.__writeResult(paraBaseline.latReprName, f)
			paraBaseline.doHouseKeeping()
			f.flush()
			

			# fixed full 
			os.environ["NBR_TYPE"]=str(0)
			os.environ["FULL_DATA"]=str(1)
			f.write("Optimal lambda fixed full: %.2f%s" %(optPDict["lambda_full_fixed"], os.linesep))	
			jointL = JointLearningSen2VecRunner(self.dbstring)
			jointL.window = optPDict["window"]
			jointL.lambda_val = optPDict["lambda-full-fixed"]
			jointL.prepareData(pd)
			jointL.runTheBaseline(rbase, latent_space_size)
			jointL.runEvaluationTask()
			self.__writeResult("%s"%jointL.latReprName, f)
			jointL.doHouseKeeping()

			
			os.environ["NBR_TYPE"]=str(1)
			os.environ["FULL_DATA"]=str(1)
			f.write("Optimal lambda fixed full: %.2f%s" %(optPDict["lambda_full_n2v"], os.linesep))	
			jointL = JointLearningSen2VecRunner(self.dbstring)
			jointL.window = optPDict["window"]
			jointL.lambda_val = optPDict["lambda-full-n2v"]
			jointL.prepareData(pd)
			jointL.runTheBaseline(rbase, latent_space_size)
			jointL.runEvaluationTask()
			self.__writeResult("%s"%jointL.latReprName, f)
			jointL.doHouseKeeping()

			os.environ["NBR_TYPE"]=str(0)
			os.environ["FULL_DATA"]=str(0)
			f.write("Optimal lambda fixed full: %.2f%s" %(optPDict["lambda_random_fixed"], os.linesep))	
			jointL = JointLearningSen2VecRunner(self.dbstring)
			jointL.window = optPDict["window"]
			jointL.lambda_val = optPDict["lambda-random-fixed"]
			jointL.prepareData(pd)
			jointL.runTheBaseline(rbase, latent_space_size)
			jointL.runEvaluationTask()
			self.__writeResult("%s"%jointL.latReprName, f)
			jointL.doHouseKeeping()

			
			os.environ["NBR_TYPE"]=str(1)
			os.environ["FULL_DATA"]=str(0)
			f.write("Optimal lambda fixed full: %.2f%s" %(optPDict["lambda_random_n2v"], os.linesep))	
			jointL = JointLearningSen2VecRunner(self.dbstring)
			jointL.window = optPDict["window"]
			jointL.lambda_val = optPDict["lambda-random-n2v"]
			jointL.prepareData(pd)
			jointL.runTheBaseline(rbase, latent_space_size)
			jointL.runEvaluationTask()
			self.__writeResult("%s"%jointL.latReprName, f)
			jointL.doHouseKeeping()