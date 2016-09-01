import os 
import logging 
from documentReader.DocumentReader import DocumentReader
from documentReader.PostgresDataRecorder   import PostgresDataRecorder
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize

from log_manager.log_config import Logger 


class ReutersReader(DocumentReader):
	""" 
	Reuters Document Reader

	"""

	def __init__(self,*args, **kwargs):
		"""
		"""
		DocumentReader.__init__(self, *args, **kwargs)
		self.dbstring = os.environ["REUTERS_DBSTRING"]
		Logger.logr.info(self.dbstring)
		self.postgres_recorder = PostgresDataRecorder(self.dbstring)
		self.folderPath = os.environ['REUTERS_PATH']

		
	def readTopic(self):
		"""
		"""
		texts = []
		categories = []
		for file_ in os.listdir(self.folderPath):
			if file_.endswith(".lc.txt"):
				category = file_.split('-')[1]
				content = open("%s%s%s" %(self.folderPath,"/",file_), 'r', 
					encoding='utf-8', errors='ignore').read()
				for topic in content.split('\n'):
					topic = topic.strip()
					if len(topic) != 0:
						texts += [topic]
						categories += [category]

		self.postgres_recorder.insertIntoTopTable(texts, categories)						
		Logger.logr.info("Topic reading complete.")

	def __recordDocument (self, document_id, doc):
		"""

		"""
		topic_names = []
		categories = []
							
		possible_categories = ["topics", "places", "people", "orgs", 
				"exchanges", "companies"] # List of possible topics

		for category in possible_categories:
			try:
				topics = doc.find(category).findAll('d')
				for topic in topics:
					topic = topic.text.strip()
					topic_names += [topic]
					categories += [category]					
			except:
				pass
		
		self.postgres_recorder.insertIntoDoc_TopTable(document_id,\
					topic_names, categories) 
	def __recordParagraphAndSentence(self, document_id, doc_content)
		"""
		"""
		paragraphs = text.split('\n\n')
		for position, paragraph in enumerate(paragraphs):
			paragraph_id = self.postgres_recorder.insertIntoParTable(paragraph)
			self.postgres_recorder.insertIntoDoc_ParTable(document_id, paragraph_id, position)
			
			sentences = sent_tokenize(paragraph)
			for sentence_position, sentence in enumerate(sentences):
				sentence_id = self.postgres_recorder.insertIntoSenTable(sentence)
				self.postgres_recorder.insertIntoPar_SenTable(paragraph_id, sentence_id, sentence_position)
						
	def readDocument(self):
		"""
		First, reading and recording the Topics
		Second, recording each document at a time	
		Third, for each document, record the lower level information 
		like: paragraph, sentences in table 
		"""
		self.readTopic() 
		
		for file_ in os.listdir(self.folderPath):
			if file_.endswith(".sgm"):
				file_content = open(self.folderPath+"/"+file, 'r', 
					encoding='utf-8', errors='ignore').read()
				soup = BeautifulSoup(file_content, "html.parser")

				for doc in soup.findAll('reuters'):
					document_id = doc['newid']
					title = doc.find('title').text
					doc_content = doc.find('text').text
					try:
						metadata = "OLDID:"+doc['oldid']+"^"+"TOPICS:"+doc['topics']+\
						"^"+"CGISPLIT:"+doc['cgisplit']+"^"+"LEWISSPLIT:"+doc['lewissplit']
					except:
						metadata = None
					
					self.postgres_recorder.insertIntoDocTable(document_id, title, \
								doc_content, file_, metadata) 


					self.__recordDocument (document_id, doc)			
					self.__recordParagraphAndSentence(doc_content)
					
					
		logging.info("Document reading complete.")
