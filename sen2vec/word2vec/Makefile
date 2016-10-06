CC = gcc
#Using -Ofast instead of -O3 might result in faster code, but is supported only by newer GCC versions
CFLAGS = -lm -pthread -Ofast -march=native -Wall -funroll-loops -Wno-unused-result

all: word2vec retrofit_word2vec_one retrofit_with_net

word2vec : word2vec.c
	$(CC) word2vec.c -o word2vec $(CFLAGS)
retrofit_word2vec_one: retrofit_word2vec.c 
	$(CC) retrofit_word2vec.c -o retrofit_word2vec_one $(CFLAGS)
retrofit_with_net: retrofit_sen2vecwithnet.c 
	$(CC) retrofit_sen2vecwithnet.c -o retrofit_with_net $(CFLAGS)


clean:
	rm -rf word2vec 
	rm -rf retrofit_word2vec
	rm -rf retrofit_sen2vecwithnet
