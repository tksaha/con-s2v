# remove unused warning out from word2vec code of google
gcc word2vec.c -o word2vec -lm -pthread -O3 -march=native -funroll-loops