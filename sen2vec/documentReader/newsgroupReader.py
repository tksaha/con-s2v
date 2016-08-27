import os 
from documentReader.DocumentReader import DocumentReader

class NewsGroupReader(DocumentReader):
	""" News Group Document Reader"""

	def __init__(self,*args, **kwargs):
		DocumentReader.__init__(self, *args, **kwargs)

	def readDocument(self, folderName): 
		#First level folder 
		first_level_folders = os.listdir(folderName)
		
		for folder in first_level_folders:
			if  not(DocumentReader._folder_is_hidden(self, folder)):
				doc_tag = folder[folder.rfind(".")+1: ]
				yield doc_tag