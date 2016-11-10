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
	
			f1 = {}
			window_opt = None #var for the optimal window
			for window in ["8", "10", "12"]:
			#for window in ["8"]:
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
			optPDict["window"] = window_opt

			Logger.logr.info("Starting Running Para2vec Baseline for Optimal Window = %s" %window_opt)
			self.postgres_recorder.truncateSummaryTable()
			paraBaseline = P2VSENTCExecutableRunner(self.dbstring)
			paraBaseline.runTheBaseline(rbase,latent_space_size, window_opt)
			paraBaseline.generateSummary(gs)
			#we need the p2v vectors created with optimal window
			paraBaseline.doHouseKeeping()

			f1 = {}
			beta_opt = None #var for the optimal beta
			for beta in ["0.3", "0.6", "0.9","1.0"]:
			#for beta in ["0.3"]:
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
				f1[beta] = self.__getF1("%s_retrofit"%n2vBaseline.latReprName)
				Logger.logr.info("F1 for %s = %s" %(beta, f1[beta]))
			beta_opt = max(f1, key=f1.get)  
			f.write("Optimal Beta is %s%s"%(beta_opt, os.linesep))
			f.write("N2V Beta f1s: %s%s" %(f1, os.linesep))
			f.flush()
			optPDict['beta'] = beta_opt
			
			f1 = {}
			joint_beta_opt = None #var for the optimal joint_beta
			for joint_beta in [0.5, 0.6, 0.7, 0.8, 0.85, 0.90, 0.95]:
				Logger.logr.info("Starting Running JointLearning Baseline for Joint-Beta = %s" %joint_beta)
				jointL = JointLearningSen2VecRunner(self.dbstring)
				jointL.jointbeta = joint_beta
				if joint_beta==0.5:
					jointL.prepareData(pd)
				jointL.runTheBaseline(rbase, 300)
				jointL.runEvaluationTask()
				jointL.doHouseKeeping()
				f1[joint_beta] = self.__getF1("%s" %jointL.latReprName)
				Logger.logr.info("F1 for %s = %s" %(joint_beta, f1[joint_beta]))
			joint_beta_opt = max(f1, key=f1.get) 
			Logger.logr.info("Optimal Joint-Beta=%s" %joint_beta_opt)
			f.write("Optimal Joint-Beta is %.2f%s"%(joint_beta_opt, os.linesep))
			f.write("JTL Joint-Beta f1s: %s%s" %(f1, os.linesep))
			f.flush()
			optPDict['joint-beta'] = joint_beta_opt
			
			f1 = {}
			fs_beta_opt = None #var for the optimal joint_beta
			for fs_beta in [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7]:
				Logger.logr.info("Starting Running FastSent Baseline for FS-Beta = %s" %fs_beta)
				frunner = FastSentVariantRunner(self.dbstring)
				frunner.fastsentbeta = fs_beta
				if fs_beta==0.3:
					frunner.prepareData(pd)
				frunner.runTheBaseline(rbase, 300)
				frunner.runEvaluationTask()
				frunner.doHouseKeeping()
				f1[fs_beta] = self.__getF1("%s" %frunner.latReprName)
				Logger.logr.info("F1 for %s = %s" %(fs_beta, f1[fs_beta]))
			fs_beta_opt = max(f1, key=f1.get) 
			Logger.logr.info("Optimal FS-Beta=%s" %fs_beta_opt)
			f.write("Optimal FS-Beta is %.2f%s"%(fs_beta_opt, os.linesep))
			f.write("FST FS-Beta f1s: %s%s" %(f1, os.linesep))
			f.flush()
			optPDict['fs-beta'] = fs_beta_opt
			
#			f1 = {}
#			alpha_opt = None #var for the optimal beta
#			for alpha in [0.3, 0.6, 0.8, 1.0]:
#			#for alpha in [0.3]:
#				Logger.logr.info("Starting Running Iterative Baseline for Alpha = %s" %alpha)
#				iterrunner = IterativeUpdateRetrofitRunner(self.dbstring)
#				iterrunner.myalpha = alpha #reinitializing myalpha
#				if alpha==0.3:
#					iterrunner.prepareData(pd)
#				iterrunner.runTheBaseline(rbase)	
#				iterrunner.runEvaluationTask()
#				iterrunner.doHouseKeeping()
#				f1[alpha] = self.__getF1("%s_weighted"%iterrunner.latReprName)	
#				Logger.logr.info("F1 for %s = %s" %(alpha, f1[alpha]))
#			alpha_opt = max(f1, key=f1.get) 
#			Logger.logr.info("Optimal Alpha=%s" %alpha_opt)
#			f.write("Optimal alpha is %.2f%s"%(alpha_opt, os.linesep))
#			f.write("ITR Alpha f1s: %s%s" %(f1, os.linesep))
#			f.flush()
#			optPDict['alpha'] = alpha_opt

#			w_f1 = {}
#			unw_f1 = {}
#			w_opt = None
#			unw_opt = None
#			for beta in [0.3, 0.6, 0.8, 1.0]:
#			#for beta in [0.3]:
#				Logger.logr.info("Starting Running Regularized Baseline for Beta = %s" %beta)
#				regs2v = RegularizedSen2VecRunner(self.dbstring)
#				regs2v.regBetaW = beta
#				regs2v.regBetaUNW = beta
#				if beta==0.3:
#					regs2v.prepareData(pd)
#				regs2v.runTheBaseline(rbase, latent_space_size)
#				regs2v.runEvaluationTask()
#				regs2v.doHouseKeeping()
#				w_f1[beta] = self.__getF1("%s_neighbor_w"%regs2v.latReprName)	
#				unw_f1[beta] = self.__getF1("%s_neighbor_unw"%regs2v.latReprName)	
#				Logger.logr.info("W_f1 for %s = %s" %(beta, w_f1[beta]))
#				Logger.logr.info("UNW_f1 for %s = %s" %(beta, unw_f1[beta]))
#			w_opt_reg = max(w_f1, key=w_f1.get)
#			unw_opt_reg = max(unw_f1, key=unw_f1.get)
#			Logger.logr.info("Optimal regBetaW=%s and regBetaUNW=%s" %(w_opt_reg, unw_opt_reg))
#			optPDict['w_opt_reg'] = w_opt_reg
#			optPDict['unw_opt_reg'] = unw_opt_reg

#			f.write("Optimal REG BetaW : %.2f%s" %(w_opt_reg, os.linesep))
#			f.write("Optimal REG BetaUNW : %.2f%s" %(unw_opt_reg, os.linesep))
#			f.write("REG BetaW f1s: %s%s" %(w_f1, os.linesep))
#			f.write("REG BetaUNW f1s: %s%s" %(unw_f1, os.linesep))
#			f.flush()

#			w_f1 = {}
#			unw_f1 = {}
#			w_opt = None
#			unw_opt = None
#			for beta in [0.3, 0.6, 0.8, 1.0]:
#			#for beta in [0.3]:
#				Logger.logr.info("Starting Running Dict Regularized Baseline for Beta = %s" %beta)
#				dictregs2v = DictRegularizedSen2VecRunner(self.dbstring)
#				dictregs2v.dictregBetaW = beta
#				dictregs2v.dictregBetaUNW = beta
#				if beta==0.3:
#					dictregs2v.prepareData(pd)
#				dictregs2v.runTheBaseline(rbase, latent_space_size)
#				dictregs2v.runEvaluationTask()
#				dictregs2v.doHouseKeeping()
#				w_f1[beta] = self.__getF1("%s_neighbor_w"%dictregs2v.latReprName)	
#				unw_f1[beta] = self.__getF1("%s_neighbor_unw"%dictregs2v.latReprName)	
#				Logger.logr.info("W_f1 for %s = %s" %(beta, w_f1[beta]))
#				Logger.logr.info("UNW_f1 for %s = %s" %(beta, unw_f1[beta]))
#			w_opt_dict_reg = max(w_f1, key=w_f1.get)
#			unw_opt_dict_reg = max(unw_f1, key=unw_f1.get)
#			Logger.logr.info("Optimal dictregBetaW=%s and dictregBetaUNW=%s" %(w_opt_dict_reg, unw_opt_dict_reg))
#			optPDict['w_opt_dict_reg'] = w_opt_dict_reg
#			optPDict['unw_opt_dict_reg'] = unw_opt_dict_reg

#			f.write("DCT BetaW: %.2f%s" %(w_opt_dict_reg, os.linesep))
#			f.write("DCT BetaUNW: %.2f%s" %(unw_opt_dict_reg, os.linesep))
#			f.write("DCT BetaW f1s: %s%s" %(w_f1, os.linesep))
#			f.write("DCT BetaUNW f1s: %s%s" %(unw_f1, os.linesep))
#			f.flush()
		return optPDict


	def _runClusteringOnValidation(self, pd, rbase, gs, dataset_name):
		############# Validation ############################		
		optPDict = {}
		with open('%s%s%s%s' %(os.environ["TRTESTFOLDER"],"/",dataset_name,"_hyperparameters_clust.txt"), 'w') as f:
			
			latent_space_size = 300
			os.environ['EVAL'] = 'VALID'
			os.environ['VALID_FOR'] = 'CLUST'
	
			adjustedMScore = {}
			window_opt = None #var for the optimal window
			for window in ["8", "10", "12"]:
			#for window in ["8"]:
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
			optPDict["window"] = window_opt

			Logger.logr.info("Starting Running Para2vec Baseline for Optimal Window = %s" %window_opt)
			self.postgres_recorder.truncateSummaryTable()
			paraBaseline = P2VSENTCExecutableRunner(self.dbstring)
			paraBaseline.runTheBaseline(rbase,latent_space_size, window_opt)
			paraBaseline.generateSummary(gs)
			paraBaseline.doHouseKeeping()

			adjustedMScore = {}
			beta_opt = None #var for the optimal beta
			for beta in ["0.3", "0.6", "0.9","1.0"]:
			#for beta in ["0.3"]:
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
				adjustedMScore[beta] = self.__getAdjustedMutulScore("%s_retrofit"%n2vBaseline.latReprName)
				Logger.logr.info("Adjusted MScore for %s = %s" %(beta, adjustedMScore[beta]))
			beta_opt = max(adjustedMScore, key=adjustedMScore.get)  
			f.write("Optimal Beta is %s%s"%(beta_opt, os.linesep))
			f.write("N2V Beta adjusted mutual score: %s%s" %(adjustedMScore, os.linesep))
			f.flush()
			optPDict["beta"] = beta_opt
			
			adjustedMScore = {}
			joint_beta_opt = None #var for the optimal beta
			for joint_beta in [0.5, 0.6, 0.7, 0.8, 0.85, 0.90, 0.95]:
				Logger.logr.info("Starting Running JointLearning Baseline for Joint-Beta = %s" %joint_beta)
				jointL = JointLearningSen2VecRunner(self.dbstring)
				jointL.jointbeta = joint_beta #reinitializing myalpha
				if 	joint_beta==0.5:
					jointL.prepareData(pd)
				jointL.runTheBaseline(rbase, 300)	
				jointL.runEvaluationTask()
				jointL.doHouseKeeping()
				adjustedMScore[joint_beta] = self.__getAdjustedMutulScore("%s"%jointL.latReprName)	
				Logger.logr.info("Adjusted Mutual Score for %s = %s" %(joint_beta, adjustedMScore[joint_beta]))
			joint_beta_opt = max(adjustedMScore, key=adjustedMScore.get) 
			Logger.logr.info("Optimal Joint-Beta=%s" %joint_beta_opt)
			f.write("Optimal Joint-Beta is %.2f%s"%(joint_beta_opt, os.linesep))
			f.write("JTL Joint-Beta adjusted mutual scores: %s%s" %(adjustedMScore, os.linesep))
			f.flush()
			optPDict["joint_beta"] = joint_beta_opt
			
			adjustedMScore = {}
			fs_beta_opt = None #var for the optimal joint_beta
			for fs_beta in [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7]:
				Logger.logr.info("Starting Running FastSent Baseline for FS-Beta = %s" %fs_beta)
				frunner = FastSentVariantRunner(self.dbstring)
				frunner.fastsentbeta = fs_beta
				if fs_beta==0.3:
					frunner.prepareData(pd)
				frunner.runTheBaseline(rbase, 300)
				frunner.runEvaluationTask()
				frunner.doHouseKeeping()
				adjustedMScore[fs_beta] = self.__getAdjustedMutulScore("%s"%frunner.latReprName)	
				Logger.logr.info("Adjusted Mutual Score for %s = %s" %(fs_beta, adjustedMScore[fs_beta]))
			fs_beta_opt = max(adjustedMScore, key=adjustedMScore.get) 
			Logger.logr.info("Optimal FS-Beta=%s" %fs_beta_opt)
			f.write("Optimal FS-Beta is %.2f%s"%(fs_beta_opt, os.linesep))
			f.write("FST FS-Beta adjusted mutual scores: %s%s" %(adjustedMScore, os.linesep))
			f.flush()
			optPDict["fs_beta"] = fs_beta_opt
			
#			adjustedMScore = {}
#			alpha_opt = None #var for the optimal beta
#			for alpha in [0.3, 0.6, 0.8, 1.0]:
#			#for alpha in [0.3]:
#				Logger.logr.info("Starting Running Iterative Baseline for Alpha = %s" %alpha)
#				iterrunner = IterativeUpdateRetrofitRunner(self.dbstring)
#				iterrunner.myalpha = alpha #reinitializing myalpha
#				if 	alpha==0.3:
#					iterrunner.prepareData(pd)
#				iterrunner.runTheBaseline(rbase)	
#				iterrunner.runEvaluationTask()
#				iterrunner.doHouseKeeping()
#				adjustedMScore[alpha] = self.__getAdjustedMutulScore("%s_weighted"%iterrunner.latReprName)	
#				Logger.logr.info("Adjusted Mutual Score for %s = %s" %(alpha, adjustedMScore[alpha]))
#			alpha_opt = max(adjustedMScore, key=adjustedMScore.get) 
#			Logger.logr.info("Optimal Alpha=%s" %alpha_opt)
#			f.write("Optimal alpha is %.2f%s"%(alpha_opt, os.linesep))
#			f.write("ITR Alpha adjusted mutual scores: %s%s" %(adjustedMScore, os.linesep))
#			f.flush()
#			optPDict["alpha"] = alpha_opt

#			w_adjusted = {}
#			unw_adjusted = {}
#			w_opt = None
#			unw_opt = None
#			for beta in [0.3, 0.6, 0.8, 1.0]:
#			#for beta in [0.3]:
#				Logger.logr.info("Starting Running Regularized Baseline for Beta = %s" %beta)
#				regs2v = RegularizedSen2VecRunner(self.dbstring)
#				regs2v.regBetaW = beta
#				regs2v.regBetaUNW = beta
#				if beta==0.3:
#					regs2v.prepareData(pd)
#				regs2v.runTheBaseline(rbase, latent_space_size)
#				regs2v.runEvaluationTask()
#				regs2v.doHouseKeeping()
#				w_adjusted[beta] = self.__getAdjustedMutulScore("%s_neighbor_w"%regs2v.latReprName)	
#				unw_adjusted[beta] = self.__getAdjustedMutulScore("%s_neighbor_unw"%regs2v.latReprName)	
#				Logger.logr.info("W_adjusted for %s = %s" %(beta, w_adjusted[beta]))
#				Logger.logr.info("UNW_adjusted for %s = %s" %(beta, unw_adjusted[beta]))
#			w_opt_reg = max(w_adjusted, key=w_adjusted.get)
#			unw_opt_reg = max(unw_adjusted, key=unw_adjusted.get)
#			Logger.logr.info("Optimal regBetaW=%s and regBetaUNW=%s" %(w_opt_reg, unw_opt_reg))
#			optPDict["w_opt_reg"] = w_opt_reg
#			optPDict["unw_opt_reg"] = unw_opt_reg

#			f.write("Optimal REG BetaW : %.2f%s" %(w_opt_reg, os.linesep))
#			f.write("Optimal REG BetaUNW : %.2f%s" %(unw_opt_reg, os.linesep))
#			f.write("REG BetaW adjusted mutual score: %s%s" %(w_adjusted, os.linesep))
#			f.write("REG BetaUNW adjusted mutual score: %s%s" %(unw_adjusted, os.linesep))
#			f.flush()

#			w_adjusted = {}
#			unw_adjusted = {}
#			w_opt = None
#			unw_opt = None
#			for beta in [0.3, 0.6, 0.8, 1.0]:
#			#for beta in [0.3]:
#				Logger.logr.info("Starting Running Dict Regularized Baseline for Beta = %s" %beta)
#				dictregs2v = DictRegularizedSen2VecRunner(self.dbstring)
#				dictregs2v.dictregBetaW = beta
#				dictregs2v.dictregBetaUNW = beta
#				if beta==0.3:
#					dictregs2v.prepareData(pd)
#				dictregs2v.runTheBaseline(rbase, latent_space_size)
#				dictregs2v.runEvaluationTask()
#				dictregs2v.doHouseKeeping()
#				w_adjusted[beta] = self.__getAdjustedMutulScore("%s_neighbor_w"%dictregs2v.latReprName)	
#				unw_adjusted[beta] = self.__getAdjustedMutulScore("%s_neighbor_unw"%dictregs2v.latReprName)	
#				Logger.logr.info("W_adjusted for %s = %s" %(beta, w_adjusted[beta]))
#				Logger.logr.info("UNW_adjusted for %s = %s" %(beta, unw_adjusted[beta]))
#			w_opt_dict_reg = max(w_adjusted, key=w_adjusted.get)
#			unw_opt_dict_reg = max(unw_adjusted, key=unw_adjusted.get)
#			Logger.logr.info("Optimal dictregBetaW=%s and dictregBetaUNW=%s" %(w_opt_dict_reg, unw_opt_dict_reg))

#			optPDict["w_opt_dict_reg"] = w_opt_dict_reg 
#			optPDict["unw_opt_dict_reg"] = unw_opt_dict_reg

#			f.write("DCT BetaW: %.2f%s" %(w_opt_dict_reg, os.linesep))
#			f.write("DCT BetaUNW: %.2f%s" %(unw_opt_dict_reg, os.linesep))
#			f.write("DCT BetaW adjusted mutual score: %s%s" %(w_adjusted, os.linesep))
#			f.write("DCT BetaUNW adjusted mutual score: %s%s" %(unw_adjusted, os.linesep))
#			f.flush()

		return optPDict


	def doTesting(self, optPDict, dataset_name, rbase, pd, gs, classification=True):
		######### Test ########################################
		os.environ["EVAL"]='TEST'
		latent_space_size = 300

		if classification==True:
			os.environ['TEST_FOR'] = 'CLASS'
		else:
			os.environ['TEST_FOR'] = 'CLUST'

		f = open('%s%s%s%s' %(os.environ["TRTESTFOLDER"],"/",dataset_name,"_testresults_%s.txt"%os.environ['VALID_FOR']), 'w') 
		niter = 5
		for i in range(0,niter):
			f.write("###### Iteration: %s ######%s" %(i, os.linesep))
			f.write("Optimal Window: %s%s" %(optPDict["window"], os.linesep))				
			self.postgres_recorder.truncateSummaryTable()
			paraBaseline = P2VSENTCExecutableRunner(self.dbstring)
			paraBaseline.runTheBaseline(rbase,latent_space_size, optPDict["window"]) 
			paraBaseline.generateSummary(gs)
			paraBaseline.runEvaluationTask()
			self.__writeResult(paraBaseline.latReprName, f)
			paraBaseline.doHouseKeeping()
			f.flush()


			f.write("Optimal Beta: %s%s" %(optPDict["beta"], os.linesep))	
			n2vBaseline = Node2VecRunner(self.dbstring)
			n2vBaseline.mybeta = optPDict["beta"]
			generate_walk = False
			n2vBaseline.runTheBaseline(rbase, latent_space_size, generate_walk)
			n2vBaseline.runEvaluationTask()
			self.__writeResult("%s"%n2vBaseline.latReprName, f)
			self.__writeResult("%s_init"%n2vBaseline.latReprName, f)
			self.__writeResult("%s_retrofit"%n2vBaseline.latReprName, f)
			n2vBaseline.doHouseKeeping()
			f.flush()


			f.write("Optimal Joint-Beta: %.2f%s" %(optPDict["joint_beta"], os.linesep))	
			jointL = JointLearningSen2VecRunner(self.dbstring)
			jointL.jointbeta = optPDict["joint_beta"]
			jointL.runTheBaseline(rbase, 300)
			jointL.runEvaluationTask()
			self.__writeResult("%s"%jointL.latReprName, f)
			jointL.doHouseKeeping()


			f.write("Optimal FS-Beta: %.2f%s" %(optPDict["fs_beta"], os.linesep))	
			frunner = FastSentVariantRunner(self.dbstring)
			frunner.fastsentbeta = optPDict["fs_beta"]
			frunner.runTheBaseline(rbase, 300)
			frunner.runEvaluationTask()
			self.__writeResult("%s"%frunner.latReprName, f)
			frunner.doHouseKeeping()


#			f.write("Optimal regBetaW: %.2f%s" %(optPDict['w_opt_reg'], os.linesep))	
#			f.write("Optimal regBetaUNW: %.2f%s" %(optPDict['unw_opt_reg'], os.linesep))	
#			regs2v = RegularizedSen2VecRunner(self.dbstring)
#			regs2v.regBetaW = optPDict['w_opt_reg']
#			regs2v.regBetaUNW = optPDict['unw_opt_reg']
#			regs2v.runTheBaseline(rbase, latent_space_size)
#			regs2v.runEvaluationTask()
#			self.__writeResult("%s_neighbor_w"%regs2v.latReprName, f)
#			self.__writeResult("%s_neighbor_unw"%regs2v.latReprName, f)
#			regs2v.doHouseKeeping()
#			f.flush()


#			f.write("Optimal dictregBetaW: %.2f%s" %(optPDict["w_opt_dict_reg"], os.linesep))	
#			f.write("Optimal dictregBetaUNW: %.2f%s" %(optPDict["unw_opt_dict_reg"], os.linesep))	
#			dictregs2v = DictRegularizedSen2VecRunner(self.dbstring)
#			dictregs2v.dictregBetaW = optPDict["w_opt_dict_reg"]
#			dictregs2v.dictregBetaUNW = optPDict["unw_opt_dict_reg"]
#			dictregs2v.runTheBaseline(rbase, latent_space_size)
#			dictregs2v.runEvaluationTask()
#			self.__writeResult("%s_neighbor_w"%dictregs2v.latReprName, f)
#			self.__writeResult("%s_neighbor_unw"%dictregs2v.latReprName, f)
#			dictregs2v.doHouseKeeping()
#			f.flush()
