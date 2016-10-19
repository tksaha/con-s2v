CC = g++ 

# Using -Ofast instead of -O3 might result in faster code, but is supported only by newer GCC versions
CFLAGS = -lm -pthread -Ofast -march=native -Wall -funroll-loops -Wno-unused-result

# for Debugging purposes, please turn on the following version of CC and CFLAGS
# CC = gcc -g -o0 
# CFLAGS = -lm -pthread  -march=native -Wall -funroll-loops -Wno-unused-result

all: bh_tsne

bh_tsne: tsne.cpp 
	$(CC) sptree.cpp tsne.cpp -o bh_tsne 

clean:
	rm -rf bh_tsne