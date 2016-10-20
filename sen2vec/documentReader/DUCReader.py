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
	
	
	def __runCombinedEvaluation(self):
		rougeInstance = Rouge()
		rPDict = rougeInstance.buildRougeParamDict()
		rPDict['-l'] = str(100)
		rPDict['-c'] = str(0.99)

		evaluation = RankingEvaluation(topics = [self.duc_topic], models = [20], systems = [1,2,3,4,5,6,7,8,9,10,11,21])
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
		for i in range(0,5):
			with open('%s%s%s' %(os.environ["TRTESTFOLDER"],"/","hyperparameters.txt"), 'w') as f:
				f.write("###### Iteration: %s ######%s" %(i, os.linesep))
				latent_space_size = 300

				diversity = False
				if self.diversity == str(1):
					diversity = True 

				# createValidationSet() Need to implement this function
				os.environ['DUC_EVAL']='VALID'
		
				recalls = {}
				window_opt = None #var for the optimal window
				for window in ["8", "10", "12"]:
					Logger.logr.info("Starting Running Para2vec Baseline for Window = %s" %window)
					self.postgres_recorder.truncateSummaryTable()
					paraBaseline = P2VSENTCExecutableRunner(self.dbstring)
					if 	window=="8":  
						paraBaseline.prepareData(pd)
					paraBaseline.runTheBaseline(rbase,latent_space_size, window)
					paraBaseline.generateSummary(gs,\
						lambda_val=self.lambda_val, diversity=diversity)
					paraBaseline.doHouseKeeping()			
					self.__runSpecificEvaluation(models = [20], systems = [2]) #Running Rouge for method_id = 2 only
					recalls[window] = self.__getRecall(method_id=2, models = [20], systems = [2])
					Logger.logr.info("Recall for %s = %s" %(window, recalls[window]))
				window_opt = max(recalls, key=recalls.get) #get the window for the max recall
				f.write("P2V Window Recalls: %s%s" %(recalls, os.linesep))
				f.flush()

				Logger.logr.info("Starting Running Para2vec Baseline for Optimal Window = %s" %window_opt)
				self.postgres_recorder.truncateSummaryTable()
				paraBaseline = P2VSENTCExecutableRunner(self.dbstring)
				paraBaseline.runTheBaseline(rbase,latent_space_size, window_opt) #we need the p2v vectors created with optimal window
				paraBaseline.doHouseKeeping()
		
				recalls = {}
				beta_opt = None #var for the optimal beta
				for beta in ["0.3", "0.6", "0.9"]:
					Logger.logr.info("Starting Running Node2vec Baseline for Beta = %s" %beta)
					self.postgres_recorder.truncateSummaryTable()
					n2vBaseline = Node2VecRunner(self.dbstring)
					n2vBaseline.mybeta = beta #reinitializing mybeta
					generate_walk = False
					if beta=="0.3":
					   n2vBaseline.prepareData(pd)
					   generate_walk = True 
					n2vBaseline.runTheBaseline(rbase, latent_space_size, generate_walk)
					n2vBaseline.generateSummary(gs, 5, "_retrofit",\
						 lambda_val=self.lambda_val, diversity=diversity)
					n2vBaseline.doHouseKeeping()
					self.__runSpecificEvaluation(models = [20], systems = [5]) #Running Rouge for method_id = 5 only
					recalls[beta] = self.__getRecall(method_id=5, models = [20], systems = [5])
					Logger.logr.info("Recall for %s = %s" %(beta, recalls[beta]))
				beta_opt = max(recalls, key=recalls.get) #get the beta for the max recall
				f.write("N2V Beta Recalls: %s%s" %(recalls, os.linesep))
				f.flush()
		
				recalls = {}
				alpha_opt = None #var for the optimal beta
				for alpha in [0.3, 0.6, 0.8, 1.0]:
					Logger.logr.info("Starting Running Iterative Baseline for Alpha = %s" %alpha)
					self.postgres_recorder.truncateSummaryTable()
					iterrunner = IterativeUpdateRetrofitRunner(self.dbstring)
					iterrunner.myalpha = alpha #reinitializing myalpha
					if alpha==0.3:
						iterrunner.prepareData(pd)
					iterrunner.runTheBaseline(rbase)
					iterrunner.generateSummary(gs, 7, "_weighted",\
						lambda_val=self.lambda_val, diversity=diversity)
					iterrunner.doHouseKeeping()
					self.__runSpecificEvaluation(models = [20], systems = [7])
					recalls[alpha] = self.__getRecall(method_id=7, models = [20], systems = [7])
					Logger.logr.info("Recall for %s = %s" %(alpha, recalls[alpha]))
				alpha_opt = max(recalls, key=recalls.get) #get the alpha for the max recall
				Logger.logr.info("Optimal Alpha=%s" %alpha_opt)
				f.write("ITR Alpha Recalls: %s%s" %(recalls, os.linesep))
				f.flush()

				w_recalls = {}
				unw_recalls = {}
				w_opt = None
				unw_opt = None
				for beta in [0.3, 0.6, 0.8, 1.0]:
					Logger.logr.info("Starting Running Regularized Baseline for Beta = %s" %beta)
					self.postgres_recorder.truncateSummaryTable()
					regs2v = RegularizedSen2VecRunner(self.dbstring)
					regs2v.regBetaW = beta
					regs2v.regBetaUNW = beta
					if beta==0.3:
						regs2v.prepareData(pd)
					regs2v.runTheBaseline(rbase, latent_space_size)
					regs2v.generateSummary(gs,9,"_neighbor_w",\
						 lambda_val=self.lambda_val, diversity=diversity)
					regs2v.generateSummary(gs,10,"_neighbor_unw",\
						 lambda_val=self.lambda_val, diversity=diversity)
					regs2v.doHouseKeeping()
					self.__runSpecificEvaluation(models = [20], systems = [9, 10])
					w_recalls[beta] = self.__getRecall(method_id=9, models = [20], systems = [9, 10])
					unw_recalls[beta] = self.__getRecall(method_id=10, models = [20], systems = [9, 10])
					Logger.logr.info("W_Recall for %s = %s" %(beta, w_recalls[beta]))
					Logger.logr.info("UNW_Recall for %s = %s" %(beta, unw_recalls[beta]))
				w_opt_reg = max(w_recalls, key=w_recalls.get)
				unw_opt_reg = max(unw_recalls, key=unw_recalls.get)
				Logger.logr.info("Optimal regBetaW=%s and regBetaUNW=%s" %(w_opt_reg, unw_opt_reg))
				f.write("REG BetaW Recalls: %s%s" %(w_recalls, os.linesep))
				f.write("REG BetaUNW Recalls: %s%s" %(unw_recalls, os.linesep))
				f.flush()

				w_recalls = {}
				unw_recalls = {}
				w_opt = None
				unw_opt = None
				for beta in [0.3, 0.6, 0.8, 1.0]:
					Logger.logr.info("Starting Running Dict Regularized Baseline for Beta = %s" %beta)
					self.postgres_recorder.truncateSummaryTable()
					dictregs2v = DictRegularizedSen2VecRunner(self.dbstring)
					dictregs2v.dictregBetaW = beta
					dictregs2v.dictregBetaUNW = beta
					if beta==0.3:
						dictregs2v.prepareData(pd)
					dictregs2v.runTheBaseline(rbase, latent_space_size)
					dictregs2v.generateSummary(gs,11,"_neighbor_w",\
						 lambda_val=self.lambda_val, diversity=diversity)
					dictregs2v.generateSummary(gs,12,"_neighbor_unw",\
						 lambda_val=self.lambda_val, diversity=diversity)
					dictregs2v.doHouseKeeping()
					self.__runSpecificEvaluation(models = [20], systems = [11, 12])
					w_recalls[beta] = self.__getRecall(method_id=11, models = [20], systems = [11, 12])
					unw_recalls[beta] = self.__getRecall(method_id=12, models = [20], systems = [11, 12])
					Logger.logr.info("W_Recall for %s = %s" %(beta, w_recalls[beta]))
					Logger.logr.info("UNW_Recall for %s = %s" %(beta, unw_recalls[beta]))
				w_opt_dict_reg = max(w_recalls, key=w_recalls.get)
				unw_opt_dict_reg = max(unw_recalls, key=unw_recalls.get)
				Logger.logr.info("Optimal dictregBetaW=%s and dictregBetaUNW=%s" %(w_opt_dict_reg, unw_opt_dict_reg))
				f.write("DCT BetaW Recalls: %s%s" %(w_recalls, os.linesep))
				f.write("DCT BetaUNW Recalls: %s%s" %(unw_recalls, os.linesep))
				f.flush()

				os.environ["DUC_EVAL"]='TEST'
				
				self.postgres_recorder.truncateSummaryTable()
				paraBaseline = P2VSENTCExecutableRunner(self.dbstring)
				paraBaseline.runTheBaseline(rbase,latent_space_size, window_opt) #we need the p2v vectors created with optimal window
				paraBaseline.generateSummary(gs,\
						lambda_val=self.lambda_val, diversity=diversity)
				paraBaseline.doHouseKeeping()

				n2vBaseline = Node2VecRunner(self.dbstring)
				n2vBaseline.mybeta = beta_opt
				n2vBaseline.runTheBaseline(rbase, latent_space_size, generate_walk)
				n2vBaseline.generateSummary(gs, 3, "",\
					 lambda_val=self.lambda_val, diversity=diversity)
				n2vBaseline.generateSummary(gs, 4, "_init",\
					 lambda_val=self.lambda_val, diversity=diversity)
				n2vBaseline.generateSummary(gs, 5, "_retrofit",\
					 lambda_val=self.lambda_val, diversity=diversity)
				n2vBaseline.doHouseKeeping()

				iterrunner = IterativeUpdateRetrofitRunner(self.dbstring)
				iterrunner.myalpha = alpha_opt #reinitializing myalpha
				iterrunner.runTheBaseline(rbase)
				iterrunner.generateSummary(gs, 6, "_unweighted",\
						lambda_val=self.lambda_val, diversity=diversity)
				iterrunner.generateSummary(gs, 7, "_weighted",\
						lambda_val=self.lambda_val, diversity=diversity)
				iterrunner.doHouseKeeping()


				regs2v = RegularizedSen2VecRunner(self.dbstring)
				regs2v.regBetaW = w_opt_reg 
				regs2v.regBetaUNW = unw_opt_reg
				regs2v.runTheBaseline(rbase, latent_space_size)
				regs2v.generateSummary(gs,8,"_neighbor_w",\
					 lambda_val=self.lambda_val, diversity=diversity)
				regs2v.generateSummary(gs,9,"_neighbor_unw",\
					 lambda_val=self.lambda_val, diversity=diversity)
				regs2v.doHouseKeeping()



				dictregs2v = DictRegularizedSen2VecRunner(self.dbstring)
				dictregs2v.dictregBetaW = w_opt_dict_reg
				dictregs2v.dictregBetaUNWW = unw_opt_dict_reg
				dictregs2v.runTheBaseline(rbase, latent_space_size)
				dictregs2v.generateSummary(gs,10,"_neighbor_w",\
					 lambda_val=self.lambda_val, diversity=diversity)
				dictregs2v.generateSummary(gs,11,"_neighbor_unw",\
					 lambda_val=self.lambda_val, diversity=diversity)
				dictregs2v.doHouseKeeping()
				
				self.__runCombinedEvaluation()

				#20__1_2_3_4_5_6_7_8_9_10_11_12_21_output_100.txt
				#20__1_2_3_4_5_6_7_8_9_10_11_12_21_output_10.txt
				f.write ("%s%s"%"#########################Running for Test (100) ###########################################", os.linesep)
				file_ = "~/Documents/sen2vec/Data/Summary/20__1_2_3_4_5_6_7_8_9_10_11_12_21_output_100.txt"
				for line in open(file_):
					f.write(line)

				f.write ("%s%s"%"#########################Running for Test (10) ###########################################", os.linesep)
				file_ = "~/Documents/sen2vec/Data/Summary/20__1_2_3_4_5_6_7_8_9_10_11_12_21_output_10.txt"
				for line in open(file_):
					f.write(line)

				f.write("%s%s"%(os.linesep, os.linesep))
				f.flush()




