import os 
import logging 
from documentReader.DocumentReader import DocumentReader
from documentReader.PostgresDataRecorder   import PostgresDataRecorder
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize

class ReutersReader(DocumentReader):
	""" 
	Reuters Document Reader

	"""

	def __init__(self,*args, **kwargs):
		"""
		"""
		DocumentReader.__init__(self, *args, **kwargs)
		self.dbstring = "reuters,naeemul,naeemul,localhost,5432"
		self.postgres_recorder = PostgresDataRecorder(self.dbstring)
		self.folderPath = "./sen2vec/documentReader/reuters21578"
		
	def readTopic(self):
		"""
		"""
		texts = []
		categories = []
		for file in os.listdir(self.folderPath):
			if file.endswith(".lc.txt"):
				category = file.split('-')[1]
				content = open(self.folderPath+"/"+file, 'r', encoding='utf-8', errors='ignore').read()
				for topic in content.split('\n'):
					topic = topic.strip()
					if len(topic) != 0:
						texts += [topic]
						categories += [category]
		self.postgres_recorder.insertIntoTopTable(texts, categories)						
		logging.info("Topic reading complete.")

	def readDocument(self):
		"""
		"""
		self.readTopic() #First, reading and recording the Topics
		
		for file in os.listdir(self.folderPath):
			if file.endswith(".sgm"):
				content = open(self.folderPath+"/"+file, 'r', encoding='utf-8', errors='ignore').read()
				soup = BeautifulSoup(content, "html.parser")

				for doc in soup.findAll('reuters'):
					try:
						document_id = doc['newid']
					except:
						document_id = None
					try:
						title = doc.find('title').text
					except:
						title = None					
					try:
						text = doc.find('text').text
					except:
						text = None					
					try:
						metadata = "OLDID:"+doc['oldid']+"^"+"TOPICS:"+doc['topics']+"^"+"CGISPLIT:"+doc['cgisplit']+"^"+"LEWISSPLIT:"+doc['lewissplit']
					except:
						metadata = None
					
					self.postgres_recorder.insertIntoDocTable(document_id, title, text, file, metadata) # Second, recording each document at a time
					
					texts = []
					categories = []
										
					possible_categories = ["topics", "places", "people", "orgs", "exchanges", "companies"] # List of possible topics
					for category in possible_categories:
						try:
							values = doc.find(category).findAll('d')
							for value in values:
								value = value.text.strip()
								texts += [value]
								categories += [category]					
						except:
							pass
					
					self.postgres_recorder.insertIntoDoc_TopTable(document_id, texts, categories) # Third, for each document, record the topic information in the document_topic table
					
					paragraphs = text.split('\n\n')
					for position, paragraph in enumerate(paragraphs):
						paragraph_id = self.postgres_recorder.insertIntoParTable(paragraph)
						self.postgres_recorder.insertIntoDoc_ParTable(document_id, paragraph_id, position)
						
						sentences = sent_tokenize(paragraph)
						for sentence_position, sentence in enumerate(sentences):
							sentence_id = self.postgres_recorder.insertIntoSenTable(sentence)
							self.postgres_recorder.insertIntoPar_SenTable(paragraph_id, sentence_id, sentence_position)
						
		logging.info("Document reading complete.")
