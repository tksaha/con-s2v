./word2vec -train sentences.txt -output vectors.txt -cbow 0 -init vec.txt -size 2 -window 10 -negative 5 -hs 0 -sample 1e-4 -threads 40 -binary 0 -iter 20 -min-count 1 -sentence-vectors 1
