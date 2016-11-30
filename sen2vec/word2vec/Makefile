
# Using -Ofast instead of -O3 might result in faster code, but is supported only by newer GCC versions
CC = gcc 
CFLAGS = -lm -pthread -Ofast -march=native -Wall -funroll-loops -Wno-unused-result -Wmaybe-uninitialized

# for Debugging purposes, please turn on the following version of CC and CFLAGS
#CC = gcc -g -o0 
#CFLAGS = -lm -pthread  -march=native -Wall -funroll-loops -Wno-unused-result

all: word2vec retrofit_word2vec_one reg_sen2vec_net  joint_learner

word2vec : word2vec.c
	$(CC) word2vec.c -o word2vec $(CFLAGS)
retrofit_word2vec_one: retrofit_word2vec.c 
	$(CC) retrofit_word2vec.c -o retrofit_word2vec_one $(CFLAGS)
reg_sen2vec_net: regularized_sen2vec_net.c 
	$(CC) regularized_sen2vec_net.c -o  reg_sen2vec_net $(CFLAGS)
# joint_word_node2vec: joint_learner_unsupervised.c 
# 	$(CC) joint_learner_unsupervised.c -o  joint_word_node2vec $(CFLAGS)
# joint_supervised: joint_learner_supervised.c 
# 	$(CC) joint_learner_supervised.c -o  joint_supervised $(CFLAGS)
joint_learner: joint_learner.c
	$(CC) joint_learner.c -o  joint_learner $(CFLAGS)



clean:
	rm -rf word2vec 
	rm -rf retrofit_word2vec_one
	rm -rf reg_sen2vec_net
	#rm -rf joint_word_node2vec
	#rm -rf joint_supervised
	rm -rf joint_learner
	
