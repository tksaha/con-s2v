import os 
import logging 
from documentReader.DocumentReader import DocumentReader
from documentReader.PostgresDataRecorder   import PostgresDataRecorder
from bs4 import BeautifulSoup


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
		
	def readTopic(self, folderPath):
		"""
		"""
		for file in os.listdir(folderPath):
			if file.endswith(".lc.txt"):
				category = file.split('-')[1]
				content = open(folderPath+"/"+file, 'r', encoding='utf-8', errors='ignore').read()
				for topic in content.split('\n'):
					topic = topic.strip()
					if len(topic) != 0:
						self.postgres_recorder.postgres_connector.insert([topic, category], "topic", ["text", "category"])
		print("Topic reading complete.")

	def readDocument(self, folderPath):
		"""
		"""
		
		self.readTopic(folderPath) #First, reading and recording the Topics
		
		for file in os.listdir(folderPath):
			if file.endswith(".sgm"):
				content = open(folderPath+"/"+file, 'r', encoding='utf-8', errors='ignore').read()
				soup = BeautifulSoup(content, "html.parser")

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
						metadata = "OLDID:"+doc['oldid']+"^"+"TOPICS:"+doc['topics']+"^"+"CGISPLIT:"+doc['cgisplit']+"^"+"LEWISSPLIT:"+doc['lewissplit']
					except:
						metadata = None
					self.postgres_recorder.postgres_connector.insert([id, title, text, file, metadata], "document", ["id", "title", "text", "file", "metadata"]) # Second, recording each document at a time
										
					categories = ["topics", "places", "people", "orgs", "exchanges", "companies"] # List of possible topics
					for category in categories:
						try:
							values = doc.find(category).findAll('d')
							for value in values:
								value = value.text.strip()								
								result = self.postgres_recorder.postgres_connector.select(["id"], ["topic"], [["text", "=", value], ["category", "=", category]], [], [])
								self.postgres_recorder.postgres_connector.insert([id, result[0][0]], "document_topic", ["document_id", "topic_id"]) # Third, for each document, record the topic information in the document_topic table
						except:
							pass
		print("Document reading complete.")
