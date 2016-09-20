#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os 
import logging 
from documentReader.DocumentReader import DocumentReader
from documentReader.PostgresDataRecorder   import PostgresDataRecorder
from log_manager.log_config import Logger
import re


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



	def __strip_newsgroup_header(self, text):
	    """
	    Given text in "news" format, strip the headers, by removing everything
	    before the first blank line.
	    """
	    _before, _blankline, after = text.partition('\n\n')
	    return after	


	def __strip_newsgroup_quoting(self, text):
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


	def __strip_newsgroup_footer(self, text):
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
	
	
	def __get_text_from_file(self, file):
		"""

		"""
		text = ""
		with open(file, encoding='utf-8', errors='ignore') as f:
			for line in f:
				text = "%s%s" %(text, line)
		return text
	
	
	def readTopic(self):
		"""
		"""
		first_level_folders = os.listdir(self.folderPath)
		first_level_folders = [name for name in first_level_folders if not (DocumentReader._folder_is_hidden(self, name))]
		first_level_folder = first_level_folders[0] #either train or test folder should be fine
		folder_names = os.listdir("%s%s%s" %(self.folderPath, "/", first_level_folder))
		topic_names = [name for name in folder_names if not (DocumentReader._folder_is_hidden(self, name))]
		categories = [name.split('.')[0] for name in topic_names]

		self.postgres_recorder.insertIntoTopTable(topic_names, categories)						
		Logger.logr.info("Topic reading complete.")
		return topic_names
	
	def readDocument(self, ld): 
		"""
		Stripping is by default inactive. For future reference it has been 
		imported from scikit-learn newsgroup reader package. 

		doc_content = self.__strip_newsgroup_header(doc_content)
		doc_content = self.__strip_newsgroup_footer(doc_content)
		doc_content = self.__strip_newsgroup_quoting(doc_content)
		"""
		if ld <= 0:
			return 0 
			
		self.postgres_recorder.trucateTables()
		self.postgres_recorder.altersequences()

		topic_names = self.readTopic()
		
		document_id = 0
		for first_level_folder in os.listdir(self.folderPath):
			if not(DocumentReader._folder_is_hidden(self, first_level_folder)):
				for topic in topic_names:					
					for file_ in os.listdir("%s%s%s%s%s" %(self.folderPath, "/", \
											first_level_folder, "/", topic)):
						doc_content = self.__get_text_from_file("%s%s%s%s%s%s%s" \
							%(self.folderPath, "/", first_level_folder, "/", topic, "/", file_))
						
						document_id += 1
						title = None						

						metadata = None
						istrain = None 
						try:
							trainortest = first_level_folder.split('-')[-1]
							metadata = "SPLIT:%s"%trainortest
							if trainortest.lower() == 'train':
								istrain = 'YES'
							else:
								istrain = 'NO'
						except:
							pass 

						self.postgres_recorder.insertIntoDocTable(document_id, title, \
									doc_content, file_, metadata) 

						category = topic.split('.')[0]
						self.postgres_recorder.insertIntoDoc_TopTable(document_id, \
									[topic], [category]) 		
						self._recordParagraphAndSentence(document_id, doc_content, self.postgres_recorder, topic, istrain)
					
					
		Logger.logr.info("Document reading complete.")
		return 1
	
	
	def runBaselines(self):
		"""
		"""
		pass
