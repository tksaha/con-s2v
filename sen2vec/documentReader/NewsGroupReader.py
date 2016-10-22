#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os 
import logging 
from documentReader.DocumentReader import DocumentReader
from documentReader.PostgresDataRecorder   import PostgresDataRecorder
from log_manager.log_config import Logger
import re
import numpy as np 
from baselineRunner.Paragraph2VecSentenceRunner  import Paragraph2VecSentenceRunner 
from baselineRunner.Node2VecRunner import Node2VecRunner
from baselineRunner.IterativeUpdateRetrofitRunner import IterativeUpdateRetrofitRunner
from baselineRunner.P2VSENTCExecutableRunner import P2VSENTCExecutableRunner

from evaluation.rankingevaluation.RankingEvaluation import RankingEvaluation

class NewsGroupReader(DocumentReader):
	""" 
	News Group Document Reader.
	"""

	def __init__(self,*args, **kwargs):
		"""
		Initialization assumes that NEWSGROUP_PATH environment is set. 
		To set in linux or mac: export NEWSGROUP_PATH=/some_directory_containing_newsgroup_data
		"""
		DocumentReader.__init__(self, *args, **kwargs)
		self.dbstring = os.environ["NEWSGROUP_DBSTRING"]
		self.postgres_recorder = PostgresDataRecorder(self.dbstring)
		self.folderPath = os.environ['NEWSGROUP_PATH']
		self.validationDict = {}
		self.topic_names = []


	def __stripNewsgroupHeader(self, text):
	    """
	    Given text in "news" format, strip the headers, by removing everything
	    before the first blank line.
	    """
	    _before, _blankline, after = text.partition('\n\n')
	    return after	


	def __stripNewsgroupQuoting(self, text):
	    """
	    Given text in "news" format, strip lines beginning with the quote
	    characters > or |, plus lines that often introduce a quoted section
	    (for example, because they contain the string 'writes:'.)
	    """
	    _QUOTE_RE = re.compile(r'(writes in|writes:|wrote:|says:|said:'
                       r'|^In article|^Quoted from|^\||^>)')
	    
	    good_lines = [line for line in text.split('\n')
	                  if not _QUOTE_RE.search(line)]
	    return '\n'.join(good_lines)


	def __stripNewsgroupFooter(self, text):
	    """
	    Given text in "news" format, attempt to remove a signature block.
	    As a rough heuristic, we assume that signatures are set apart by either
	    a blank line or a line made of hyphens, and that it is the last such line
	    in the file (disregarding blank lines at the end).
	    """
	    lines = text.strip().split('\n')
	    for line_num in range(len(lines) - 1, -1, -1):
	        line = lines[line_num]
	        if line.strip().strip('-') == '':
	            break

	    if line_num > 0:
	        return '\n'.join(lines[:line_num])
	    else:
	        return text
	
	
	def readTopic(self):
		"""
		http://pythoncentral.io/how-to-traverse-a-directory-tree-in-python-guide-to-os-walk/
		"""

		rootDir = "%s/20news-bydate-train" %self.folderPath
		return self._getTopics(rootDir)

	def stripDocContent(self, doc_content):
		doc_content = self.__stripNewsgroupHeader(doc_content)
		doc_content = self.__stripNewsgroupFooter(doc_content)
		return self.__stripNewsgroupQuoting(doc_content)

	def __createValidationSet(self, document_ids):

		total_doc = len(document_ids)
		nvalid_doc = float(total_doc * 0.20)

		np.random.seed(2000)
		valid_list = np.random.choice(document_ids, nvalid_doc, replace=False).tolist()

		for id_ in valid_list:
			self.validationDict[id_] = 1

	def __readAPass(self, load=0):
		if load == 0:
			self.topic_names = self.readTopic()

		train_doc_ids = []
		document_id = 0
		for first_level_folder in os.listdir(self.folderPath):
			if not(DocumentReader._folderISHidden(self, first_level_folder)):
				for topic in self.topic_names:					
					if topic not in ['talk.politics.mideast', 'comp.graphics', 'soc.religion.christian', 'rec.autos', 'sci.space', 'talk.politics.guns', 'rec.sport.baseball', 'sci.med']:
						continue
					for file_ in os.listdir("%s%s%s%s%s" %(self.folderPath, "/", \
											first_level_folder, "/", topic)):
						doc_content = self._getTextFromFile("%s%s%s%s%s%s%s" \
							%(self.folderPath, "/", first_level_folder, "/", topic, "/", file_))
						
						doc_content = self.stripDocContent(doc_content)

						document_id += 1
						title, metadata, istrain = None, None, None
						try:
							trainortest = first_level_folder.split('-')[-1]
							metadata = "SPLIT:%s"%trainortest
							istrain = 'YES' if (trainortest.lower() == 'train') else 'NO'
						except:
							Logger.logr.info("NO MetaData or Train Test Tag")

						if istrain=='YES':
							train_doc_ids.append(document_id)

						if document_id in self.validationDict:
							istrain = 'VALID'

							
						if load ==1:
							self.postgres_recorder.insertIntoDocTable(document_id, title, \
										doc_content, file_, metadata) 
							category = topic.split('.')[0]
							self.postgres_recorder.insertIntoDocTopTable(document_id, \
										[topic], [category]) 		
							self._recordParagraphAndSentence(document_id, doc_content, self.postgres_recorder, topic, istrain)
					
					
		Logger.logr.info("A pass of the document reading complete.")
		return  train_doc_ids
	
	def readDocument(self, ld): 
		"""
		Stripping is by default inactive. For future reference it has been 
		imported from scikit-learn newsgroup reader package. 

		
		"""
		if ld <= 0: return 0 			
		self.postgres_recorder.trucateTables()
		self.postgres_recorder.alterSequences()

		train_doc_ids = self.__readAPass(0)
		self.__createValidationSet(train_doc_ids)
		self.__readAPass(1)
		return 1
	
	
	def runBaselines(self, pd, rbase, gs):
		"""
		"""
		optDict = self._runClassificationOnValidation(pd, rbase, gs,"news")
		self.doTesting(optDict, "news", rbase, pd, gs, True)
		optDict = self._runClusteringOnValidation(pd, rbase, gs, "news")
		self.doTesting(optDict, "news", rbase, pd, gs, False)
