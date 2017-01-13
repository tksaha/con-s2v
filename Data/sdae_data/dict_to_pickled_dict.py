import gensim 
import os 
import sys 
import pickle
from gensim.models import Word2Vec

word_dict = {} 

model = Word2Vec.load_word2vec_format(sys.argv[1], binary=False)


line_n = 0 

for line in open(sys.argv[1]):
	if line_n == 0:
		line_n +=1 
		continue 

	key = line.strip().split()[0]
	word_dict[key] = model[key]


print (len(word_dict))
pickle.dump(word_dict, open( "glove.pkl", "wb" ) )