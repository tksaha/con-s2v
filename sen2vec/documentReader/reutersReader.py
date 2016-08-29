import os 
from documentReader.DocumentReader import DocumentReader

from bs4 import BeautifulSoup

class ReutersReader(DocumentReader):
	""" Reuters Document Reader"""

	def __init__(self,*args, **kwargs):
		DocumentReader.__init__(self, *args, **kwargs)
		
#	def recordEntity()
#		for file in os.listdir(folderPath):
#			if file.endswith(".lc.txt"):
#				content = open(folderPath+"/"+file, 'r', encoding='utf-8', errors='ignore').read()

	def readDocument(self, folderPath):
		for file in os.listdir(folderPath):
			if file.endswith(".sgm"):
				content = open(folderPath+"/"+file, 'r', encoding='utf-8', errors='ignore').read()
				soup = BeautifulSoup(content)
				
				for doc in soup.findAll('reuters'):
					try:
						id = doc['newid']
					except:
						id = None
					try:
						title = doc.find('title').text
					except:
						title = None
					try:
						text = doc.find('text').text
					except:
						text = None
					try:
						time = doc.find('date').text
					except:
						time = None
					try:
						metadata = "OLDID:"+doc['oldid']+"^"+"TOPICS:"+doc['topics']+"^"+"CGISPLIT:"+doc['cgisplit']+"^"+"LEWISSPLIT:"+doc['lewissplit']
					except:
						metadata = None
					print(id, title, text, time, metadata)
					
