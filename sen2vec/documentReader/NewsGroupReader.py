#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os 
import re 
from documentReader.DocumentReader import DocumentReader
import logging 


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
		self.newsfolder = os.environ['NEWSGROUP_PATH']

	def __strip_newsgroup_header(text):
	    """
	    Given text in "news" format, strip the headers, by removing everything
	    before the first blank line.
	    """
	    _before, _blankline, after = text.partition('\n\n')
	    return after


	_QUOTE_RE = re.compile(r'(writes in|writes:|wrote:|says:|said:'
                       r'|^In article|^Quoted from|^\||^>)')


	def __strip_newsgroup_quoting(text):
	    """
	    Given text in "news" format, strip lines beginning with the quote
	    characters > or |, plus lines that often introduce a quoted section
	    (for example, because they contain the string 'writes:'.)
	    """
	    good_lines = [line for line in text.split('\n')
	                  if not _QUOTE_RE.search(line)]
	    return '\n'.join(good_lines)


	def __strip_newsgroup_footer(text):
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

	def readDocument(self): 
		"""

		"""
		folderName = self.newsfolder
		first_level_folders = os.listdir(self.folderName)
		
		for tag_folder in first_level_folders:
			if  not(DocumentReader._folder_is_hidden(self, tag_folder)):
				doc_tag = tag_folder[tag_folder.rfind(".")+1: ]
				topic_folders = "%s/%s" %(folderName,tag_folder)
				for topic_folder in os.listdir(topic_folders):
					if  not(DocumentReader._folder_is_hidden(self, topic_folder)):
						topic = topic_folder
						docs_folders = "%s/%s" %(topic_folders,topic_folder)
						for docs in os.listdir(docs_folders):
							docs_file = "%s/%s" %(docs_folders,docs)
							if os.path.isfile(docs_file):
							   doc_id = docs 
							   # Get the text
							   yield doc_id, self.__get_text_from_file(docs_file) 