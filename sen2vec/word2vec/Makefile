CC = gcc
#Using -Ofast instead of -O3 might result in faster code, but is supported only by newer GCC versions
CFLAGS = -lm -pthread -Ofast -march=native -Wall -funroll-loops -Wno-unused-result

all: word2vec retrofit_word2vec_one unw_reg_sen2vec wgt_reg_sen2vec

word2vec : word2vec.c
	$(CC) word2vec.c -o word2vec $(CFLAGS)
retrofit_word2vec_one: retrofit_word2vec.c 
	$(CC) retrofit_word2vec.c -o retrofit_word2vec_one $(CFLAGS)
unw_reg_sen2vec: unweighted_regularized_sen2vecwithnet.c 
	$(CC) unweighted_regularized_sen2vecwithnet.c -o  unw_reg_sen2vec $(CFLAGS)
wgt_reg_sen2vec: weighted_regularized_sen2vecwithnet.c 
	$(CC) weighted_regularized_sen2vecwithnet.c -o  wgt_reg_sen2vec $(CFLAGS)



clean:
	rm -rf word2vec 
	rm -rf retrofit_word2vec_one
	rm -rf unw_reg_sen2vec
	rm -rf wgt_reg_sen2vec
