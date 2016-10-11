CC = gcc -g -o0
#Using -Ofast instead of -O3 might result in faster code, but is supported only by newer GCC versions
CFLAGS = -lm -pthread -Ofast -march=native -Wall -funroll-loops -Wno-unused-result

all: word2vec retrofit_word2vec_one reg_sen2vec_net

word2vec : word2vec.c
	$(CC) word2vec.c -o word2vec $(CFLAGS)
retrofit_word2vec_one: retrofit_word2vec.c 
	$(CC) retrofit_word2vec.c -o retrofit_word2vec_one $(CFLAGS)
reg_sen2vec_net: regularized_sen2vec_net.c 
	$(CC) regularized_sen2vec_net.c -o  reg_sen2vec_net $(CFLAGS)



clean:
	rm -rf word2vec 
	rm -rf retrofit_word2vec_one
	rm -rf reg_sen2vec_net
	
