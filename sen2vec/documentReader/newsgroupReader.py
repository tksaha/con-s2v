from documentReader.DocumentReader import DocumentReader

class NewsGroupReader(DocumentReader):
	""" News Group Document Reader"""

	def __init__(self,*args, **kwargs):
		DocumentReader.__init__(self, *args, **kwargs)

	def readDocument(self): 
		return "Implemented "