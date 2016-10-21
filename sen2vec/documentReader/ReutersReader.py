#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os 
import operator
import numpy as np 
from documentReader.DocumentReader import DocumentReader
from documentReader.PostgresDataRecorder   import PostgresDataRecorder
from bs4 import BeautifulSoup
from log_manager.log_config import Logger 
from baselineRunner.Paragraph2VecSentenceRunner  import Paragraph2VecSentenceRunner 
from baselineRunner.Node2VecRunner import Node2VecRunner
from baselineRunner.IterativeUpdateRetrofitRunner import IterativeUpdateRetrofitRunner
from baselineRunner.P2VSENTCExecutableRunner import P2VSENTCExecutableRunner
from baselineRunner.RegularizedSen2VecRunner import RegularizedSen2VecRunner
from baselineRunner.DictRegularizedSen2VecRunner import DictRegularizedSen2VecRunner

from evaluation.rankingevaluation.RankingEvaluation import RankingEvaluation 


class ReutersReader(DocumentReader):
	""" 
	Reuters Document Reader

	"""

	def __init__(self,*args, **kwargs):
		"""
		It reads he environment variable and initializes the 
		base class. 
		"""
		DocumentReader.__init__(self, *args, **kwargs)
		self.dbstring = os.environ["REUTERS_DBSTRING"]
		self.postgres_recorder = PostgresDataRecorder(self.dbstring)
		self.folderPath = os.environ['REUTERS_PATH']
		self.validationDict = {}


	def __recordDocumentTopic (self, document_id, doc):
		"""

		"""
		topic_names = []
		categories = []
							
		possible_categories = ["topics", "places", "people", "orgs", 
				"exchanges", "companies"] 

		for category in possible_categories:
			try:
				topics = doc.find(category).findAll('d')
				for topic in topics:
					topic = topic.text.strip()
					topic_names += [topic]
					categories += [category]					
			except:
				pass
		
		self.postgres_recorder.insertIntoDocTopTable(document_id,\
					topic_names, categories) 
	

	def readTopic(self):
		"""
		"""
		topic_names = []
		categories = []
		for file_ in os.listdir(self.folderPath):
			if file_.endswith(".lc.txt"):
				category = file_.split('-')[1]
				content = open("%s%s%s" %(self.folderPath,"/",file_), 'r', 
					encoding='utf-8', errors='ignore').read()
				for topic in content.split(os.linesep):
					topic = topic.strip()
					if len(topic) != 0:
						topic_names += [topic]
						categories += [category]

		self.postgres_recorder.insertIntoTopTable(topic_names, categories)						
		Logger.logr.info("Topic reading complete.")



	def _getTopic(self, document_id, doc):
		"""
		Interested topic: acq, money-fx, crude, trade, interest. 
		A topic can be one of the interested topic. A topic 
		is assigned based on the order if multiple interested topics 
		are assigned for a particular document. We take top-10 
		frequent topics mentioned in "Text Categorization with support 
		vector machines: Learning with many relevant features."
		"""
		interested_topic_list = ['earn', 'acq', 'money-fx', 'grain', 'crude', 'trade'\
			,'interest', 'ship', 'wheat', 'corn']

		topics = doc.find("topics").findAll('d')
		for topic in topics: 
			topic = topic.text.strip() 
			if topic in interested_topic_list: 
				return topic; 

		return "other"

	def __createValidationSet(self, document_ids):

		total_doc = len(document_ids)
		nvalid_doc = float(total_doc * 0.20)

		np.random.seed(2000)
		valid_list = np.random.choice(document_ids, nvalid_doc, replace=False).tolist()

		for id_ in valid_list:
			self.validationDict[id_] = 1

	def __readAPass(self,load):
		self.postgres_recorder.trucateTables()
		self.postgres_recorder.alterSequences()
		self.readTopic() 
		
		train_doc_ids = []
		for file_ in os.listdir(self.folderPath):
			if file_.endswith(".sgm"):
				file_content = self._getTextFromFile("%s%s%s" %(self.folderPath,"/",file_))
				soup = BeautifulSoup(file_content, "html.parser")
				for doc in soup.findAll('reuters'):
					document_id = doc['newid']	
					title = doc.find('title').text if doc.find('title') \
								is not None else None 
					doc_content = doc.find('text').text if doc.find('text')\
							 is not None else None 
					try:
						metadata = "OLDID:"+doc['oldid']+"^"+"TOPICS:"+doc['topics']+\
						"^"+"CGISPLIT:"+doc['cgisplit']+"^"+"LEWISSPLIT:"+doc['lewissplit']

						if doc['lewissplit'] == "NOT-USED" or doc['topics'] == "NO"\
						or doc['topics'] == "BYPASS" :
							Logger.logr.info("SKipping because of ModApte Split")
							continue
					except:
						metadata = None
						continue 
					topic = self._getTopic(document_id, doc)					

					#if topic in ['wheat', 'corn', 'other']:
					 	#continue
					if topic not in ['acq','crude']:
						continue				
						
					istrain = 'YES' if doc['lewissplit'].lower() == 'train' else 'NO'
					if document_id in self.validationDict:
						istrain ='VALID'

					if istrain == 'YES':
						train_doc_ids.append(document_id)

					if  load==0:
						continue 
					self.postgres_recorder.insertIntoDocTable(document_id, title, \
								doc_content, file_, metadata)
					self.__recordDocumentTopic(document_id, doc)
					self._recordParagraphAndSentence(document_id, doc_content, self.postgres_recorder,topic, istrain)
					
		return train_doc_ids			
		Logger.logr.info("[Pass 1] Document reading complete.")

	def readDocument(self, ld):

		"""
		First, reading and recording the Topics. Second, recording each document at a time	
		Third, for each document, record the lower level information 
		like: paragraph, sentences in table 
		"""
		if ld <= 0: return 0 
		train_doc_ids = self.__readAPass(0)
		self.__createValidationSet(train_doc_ids)
		self.__readAPass(1)
		return 1
	

	def __getF1(self, latreprName):
		"""
		"""
		file_ = os.path.join(os.environ["TRTESTFOLDER"], "%s_raweval_2.txt"%latreprName)
		for line in open(file_):
			if "avg" in line:
				line_elems = line.strip().split()
				f1 = float(line_elems[5])
				return f1 


	def __runClassificationEvaluation(self, pd, rbase, gs):
		############# Validation ############################		
		with open('%s%s%s%s' %(os.environ["TRTESTFOLDER"],"/","reuter","_hyperparameters.txt"), 'w') as f:
			
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
				n2vBaseline.doHouseKeeping()
				f1[beta_opt] = self.__getF1("%s_retrofit"%n2vBaseline.latReprName)
				Logger.logr.info("F1 for %s = %s" %(beta, f1[window]))
			beta_opt = max(f1, key=f1.get)  
			f.write("Optimal Beta is %s%s"%(beta_opt, os.linesep))
			f.write("N2V Beta Recalls: %s%s" %(f1, os.linesep))
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
				iterrunner.doHouseKeeping()
				f1[beta_opt] = self.__getF1("%s_weighted"%iterrunner.latReprName)	
				Logger.logr.info("F1 for %s = %s" %(alpha, f1[alpha]))
			alpha_opt = max(f1, key=f1.get) #get the alpha for the max recall
			Logger.logr.info("Optimal Alpha=%s" %alpha_opt)
			f.write("Optimal alpha is %.2f%s"%(alpha_opt, os.linesep))
			f.write("ITR Alpha Recalls: %s%s" %(recalls, os.linesep))
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
			f.write("REG BetaW Recalls: %s%s" %(w_recalls, os.linesep))
			f.write("REG BetaUNW Recalls: %s%s" %(unw_recalls, os.linesep))
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
				dictregs2v.doHouseKeeping()
				w_recalls[beta] = self.__getF1("%s_neighbor_w"%dictregs2v.latReprName)	
				unw_recalls[beta] = self.__getF1("%s_neighbor_unw"%dictregs2v.latReprName)	
				Logger.logr.info("W_f1 for %s = %s" %(beta, w_f1[beta]))
				Logger.logr.info("UNW_f1 for %s = %s" %(beta, unw_f1[beta]))
			w_opt_dict_reg = max(W_f1, key=w_f1.get)
			unw_opt_dict_reg = max(unw_f1, key=unw_f1.get)
			Logger.logr.info("Optimal dictregBetaW=%s and dictregBetaUNW=%s" %(w_opt_dict_reg, unw_opt_dict_reg))

			f.write("DCT BetaW: %.2f%s" %(w_opt_dict_reg, os.linesep))
			f.write("DCT BetaUNW: %.2f%s" %(unw_opt_dict_reg, os.linesep))
			f.write("DCT BetaW Recalls: %s%s" %(w_recalls, os.linesep))
			f.write("DCT BetaUNW Recalls: %s%s" %(unw_recalls, os.linesep))
			f.flush()


	def doTesting(self, f, ):
		######### Test ########################################
		os.environ["EVAL"]='TEST'

		niter = 5
		for i in range(0,niter):
			f.write("###### Iteration: %s ######%s" %(i, os.linesep))
			f.write("Optimal Window: %s%s" %(window_opt, os.linesep))				
			self.postgres_recorder.truncateSummaryTable()
			paraBaseline = P2VSENTCExecutableRunner(self.dbstring)
			paraBaseline.runTheBaseline(rbase,latent_space_size, window_opt) 
			#we need the p2v vectors created with optimal window
			paraBaseline.generateSummary(gs)
			paraBaseline.doHouseKeeping()
			f.flush()


			f.write("Optimal Beta: %s%s" %(beta_opt, os.linesep))	
			n2vBaseline = Node2VecRunner(self.dbstring)
			n2vBaseline.mybeta = beta_opt
			generate_walk = False
			n2vBaseline.runTheBaseline(rbase, latent_space_size, generate_walk)
			n2vBaseline.doHouseKeeping()
			f.flush()


			f.write("Optimal alpha: %.2f%s" %(alpha_opt, os.linesep))	
			iterrunner = IterativeUpdateRetrofitRunner(self.dbstring)
			iterrunner.myalpha = alpha_opt #reinitializing myalpha
			iterrunner.runTheBaseline(rbase)
			iterrunner.doHouseKeeping()


			f.write("Optimal regBetaW: %.2f%s" %(w_opt_reg, os.linesep))	
			f.write("Optimal regBetaUNW: %.2f%s" %(unw_opt_reg, os.linesep))	
			regs2v = RegularizedSen2VecRunner(self.dbstring)
			regs2v.regBetaW = w_opt_reg 
			regs2v.regBetaUNW = unw_opt_reg
			regs2v.runTheBaseline(rbase, latent_space_size)
			regs2v.doHouseKeeping()
			f.flush()


			f.write("Optimal regBetaW: %.2f%s" %(w_opt_dict_reg, os.linesep))	
			f.write("Optimal regBetaUNW: %.2f%s" %(unw_opt_dict_reg, os.linesep))	
			dictregs2v = DictRegularizedSen2VecRunner(self.dbstring)
			dictregs2v.dictregBetaW = w_opt_dict_reg
			dictregs2v.dictregBetaUNW = unw_opt_dict_reg
			dictregs2v.runTheBaseline(rbase, latent_space_size)
			dictregs2v.doHouseKeeping()
			f.flush()
				
				

	def runBaselines(self, pd, rbase, gs):
		"""
		"""
		self.__runClassificationEvaluation(pd, rbase, gs)
		self.doTesting()
	






