#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os 
import logging 
from documentReader.DocumentReader import DocumentReader
from documentReader.PostgresDataRecorder   import PostgresDataRecorder
from log_manager.log_config import Logger
import re
from baselineRunner.Paragraph2VecSentenceRunner  import Paragraph2VecSentenceRunner
from baselineRunner.Node2VecRunner  import Node2VecRunner

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
		doc_content = self.__strip_newsgroup_header(doc_content)
		doc_content = self.__strip_newsgroup_footer(doc_content)
		return self.__strip_newsgroup_quoting(doc_content)

	
	def readDocument(self, ld): 
		"""
		Stripping is by default inactive. For future reference it has been 
		imported from scikit-learn newsgroup reader package. 

		
		"""
		if ld <= 0: return 0 			
		self.postgres_recorder.trucateTables()
		self.postgres_recorder.alterSequences()
		topic_names = self.readTopic()
		

		document_id = 0
		for first_level_folder in os.listdir(self.folderPath):
			if not(DocumentReader._folderISHidden(self, first_level_folder)):
				for topic in topic_names:					
					for file_ in os.listdir("%s%s%s%s%s" %(self.folderPath, "/", \
											first_level_folder, "/", topic)):
						doc_content = self._getTextFromFile("%s%s%s%s%s%s%s" \
							%(self.folderPath, "/", first_level_folder, "/", topic, "/", file_))
						
						document_id += 1
						title, metadata, istrain = None, None, None 						
						try:
							trainortest = first_level_folder.split('-')[-1]
							metadata = "SPLIT:%s"%trainortest
							istrain = 'YES' if (trainortest.lower() == 'train') else 'NO'
						except:
							Logger.logr.info("NO MetaData or Train Test Tag")
							
						self.postgres_recorder.insertIntoDocTable(document_id, title, \
									doc_content, file_, metadata) 
						category = topic.split('.')[0]
						self.postgres_recorder.insertIntoDocTopTable(document_id, \
									[topic], [category]) 		
						self._recordParagraphAndSentence(document_id, doc_content, self.postgres_recorder, topic, istrain)
					
					
		Logger.logr.info("Document reading complete.")
		return 1
	
	
	def runBaselines(self):
		"""
		"""
		latent_space_size = 300
		Logger.logr.info("Starting Running Para2vec Baseline")
		paraBaseline = Paragraph2VecSentenceRunner(self.dbstring)
		paraBaseline.prepareData()
		paraBaseline.runTheBaseline(latent_space_size)

		Logger.logr.info("Starting Running Node2vec Baseline")
		n2vBaseline = Node2VecRunner(self.dbstring)
		n2vBaseline.prepareData()

		paraBaseline.runEvaluationTask()
		paraBaseline.runClassificationTask()
		
#		n2vBaseline.runTheBaseline(latent_space_size)

#		Logger.logr.info("Starting Running Iterative Update Method")
#		iterUdateBaseline = IterativeUpdateRetrofitRunner(self.dbstring)
#		iterUdateBaseline.prepareData()
#		iterUdateBaseline.runTheBaseline()
		pass
