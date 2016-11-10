//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40

const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary

typedef float real;                    // Precision of float numbers

struct vocab_word {
  long long cn;
  int *point;
  char *word, *code, codelen;
};

struct n2vword {
  long long cn;
  long long index; 
};


char train_file[MAX_STRING], output_file[MAX_STRING];
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];
char initfromFile[MAX_STRING];
char neighborFile[MAX_STRING];
char labelFile[MAX_STRING];

struct vocab_word *vocab;
struct n2vword *n2vvocab; 

long long  max_neighbors;
int binary = 0, cbow = 1, debug_mode = 2, window = 5, min_count = 5, num_threads = 12, min_reduce = 1; 
int *vocab_hash;
long long vocab_max_size = 1000, vocab_size = 0, layer1_size = 100, sentence_vectors = 0;
long long train_words = 0, word_count_actual = 0, iter = 5, file_size = 0, classes = 0;
real alpha = 0.025, starting_alpha, sample = 1e-3, beta = 1.0, beta_label = 0.0;

// All synapses and exponential table
// Initembedding for retrofitting
real *syn0, *syn0temp, *temp,  *syn1, *syn1neg, *initembed, *expTable, *templabel, *syn1label; 
long long  *nbrs;
long long  *label_sent; 
int nlabels; 


clock_t start;

int hs = 0, negative = 5;
const int table_size = 1e8;
int *table;
int *table_n2v; 


/* http://qwone.com/~jason/writing/unigram.pdf
The chance of observing a given document is simply
the product of the word probabilities. To calculate the chance of observing a
given set of word frequencies, we must count all the possible orderings that
achieve that set of frequencies. 

Used for negative sampling only.
*/
void InitUnigramTable() {
  int a, i;
  long long train_words_pow = 0;
  real d1, power = 0.75;
  table = (int *)malloc(table_size * sizeof(int));
  for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, power);
  i = 0;
  d1 = pow(vocab[i].cn, power) / (real)train_words_pow;
  for (a = 0; a < table_size; a++) {
    table[a] = i;
    if (a / (real)table_size > d1) {
      i++;
      d1 += pow(vocab[i].cn, power) / (real)train_words_pow;
    }
    if (i >= vocab_size) i = vocab_size - 1;
  }
}

// Tanay: I observe that, the probability is not normalized. 
// The use of long long data type for train_word_pow
// makes it un normailzed===> It does not sum up to 1.0
void InitUnigramTableN2V() {
  int a, i;
  long long train_words_pow = 0;
  real d1, power = 0.75;
  table_n2v = (int *)malloc(table_size * sizeof(int));
  for (a = 0; a < vocab_size; a++) train_words_pow += pow(n2vvocab[a].cn, power);
  //printf("%lld\n",train_words_pow);
  i = 0;
  d1 = pow(n2vvocab[i].cn, power) / (real)train_words_pow;
  printf("%d --- %lld -- %lf\n", i,n2vvocab[i].index, d1);

  for (a = 0; a < table_size; a++) {
    table_n2v[a] = n2vvocab[i].index;
    if (a / (real)table_size > d1) {
      i++;
      d1 += pow(n2vvocab[i].cn, power) / (real)train_words_pow;
    }
    //printf("%d --- %lld -- %lf\n", a,n2vvocab[i].index, d1);
    if (i >= vocab_size) i = vocab_size - 1;
  }
}



// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void ReadWord(char *word, FILE *fin) {
  int a = 0, ch;
  while (!feof(fin)) {
    ch = fgetc(fin);
    if (ch == 13) continue;
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      if (a > 0) {
        if (ch == '\n') ungetc(ch, fin);
        break;
      }
      if (ch == '\n') {
        strcpy(word, (char *)"</s>");
        return;
      } else continue;
    }
    word[a] = ch;
    a++;
    if (a >= MAX_STRING - 1) a--;   // Truncate too long words
  }
  word[a] = 0;
}

// Returns hash value of a word
int GetWordHash(char *word) {
  unsigned long long a, hash = 0;
  for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
  hash = hash % vocab_hash_size;
  return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(char *word) {
  unsigned int hash = GetWordHash(word);
  while (1) {
    if (vocab_hash[hash] == -1) return -1;
    if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
    hash = (hash + 1) % vocab_hash_size;
  }
  return -1;
}

// Reads a word and returns its index in the vocabulary
int ReadWordIndex(FILE *fin) {
  char word[MAX_STRING];
  ReadWord(word, fin);
  if (feof(fin)) return -1;
  return SearchVocab(word);
}



// Adds a word to the vocabulary
int AddWordToVocab(char *word) {
  unsigned int hash, length = strlen(word) + 1;
  if (length > MAX_STRING) length = MAX_STRING;
  vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
  strcpy(vocab[vocab_size].word, word);
  vocab[vocab_size].cn = 0;
  vocab_size++;
  // Reallocate memory if needed
  if (vocab_size + 2 >= vocab_max_size) {
    vocab_max_size += 1000;
    vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
  }
  hash = GetWordHash(word);
  while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
  vocab_hash[hash] = vocab_size - 1;
  return vocab_size - 1;
}

// Used later for sorting by word counts
int VocabCompare(const void *a, const void *b) {
    return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

// Used later for sorting by word counts
int VocabCompareN2V(const void *a, const void *b) {
    return ((struct n2vword *)b)->cn - ((struct n2vword *)a)->cn;
}


// https://github.com/dav/word2vec/blob/master/src/word2vec.c
void DestroyVocab() {
  int a;

  for (a = 0; a < vocab_size; a++) {
    if (vocab[a].word != NULL) {
      free(vocab[a].word);
    }
    if (vocab[a].code != NULL) {
      free(vocab[a].code);
    }
    if (vocab[a].point != NULL) {
      free(vocab[a].point);
    }
  }
  free(vocab[vocab_size].word);
  free(vocab);

  // free n2vvocab 
  free(n2vvocab);
}


// Sorts the vocabulary by frequency using word counts
void SortVocab() {
  int a, size;
  unsigned int hash;
  // Sort the vocabulary and keep </s> at the first position
  qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  size = vocab_size;
  train_words = 0;
  for (a = 0; a < size; a++) {
    // Words occuring less than min_count times will be discarded from the vocab
    if (vocab[a].cn < min_count) {
      vocab_size--;
      free(vocab[vocab_size].word);
    } else {
      // Hash will be re-computed, as after the sorting it is not actual
      hash=GetWordHash(vocab[a].word);
      while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
      vocab_hash[hash] = a;
      train_words += vocab[a].cn;
    }
  }
  vocab = (struct vocab_word *)realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));
  // Allocate memory for the binary tree construction
  for (a = 0; a < vocab_size; a++) {
    vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
    vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
  }
}




// Reduces the vocabulary by removing infrequent tokens
void ReduceVocab() {
  int a, b = 0;
  unsigned int hash;
  for (a = 0; a < vocab_size; a++) if (vocab[a].cn > min_reduce) {
    vocab[b].cn = vocab[a].cn;
    vocab[b].word = vocab[a].word;
    b++;
  } else free(vocab[a].word);
  vocab_size = b;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  for (a = 0; a < vocab_size; a++) {
    // Hash will be re-computed, as it is not actual
    hash = GetWordHash(vocab[a].word);
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = a;
  }
  fflush(stdout);
  min_reduce++;
}

// Create binary Huffman tree using the word counts
// Frequent words will have short uniqe binary codes
void CreateBinaryTree() {
  long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];
  char code[MAX_CODE_LENGTH];
  long long *count = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  long long *binary = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  long long *parent_node = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  for (a = 0; a < vocab_size; a++) count[a] = vocab[a].cn;
  for (a = vocab_size; a < vocab_size * 2; a++) count[a] = 1e15;
  pos1 = vocab_size - 1;
  pos2 = vocab_size;
  // Following algorithm constructs the Huffman tree by adding one node at a time
  for (a = 0; a < vocab_size - 1; a++) {
    // First, find two smallest nodes 'min1, min2'
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min1i = pos1;
        pos1--;
      } else {
        min1i = pos2;
        pos2++;
      }
    } else {
      min1i = pos2;
      pos2++;
    }
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min2i = pos1;
        pos1--;
      } else {
        min2i = pos2;
        pos2++;
      }
    } else {
      min2i = pos2;
      pos2++;
    }
    count[vocab_size + a] = count[min1i] + count[min2i];
    parent_node[min1i] = vocab_size + a;
    parent_node[min2i] = vocab_size + a;
    binary[min2i] = 1;
  }
  // Now assign binary code to each vocabulary word
  for (a = 0; a < vocab_size; a++) {
    b = a;
    i = 0;
    while (1) {
      code[i] = binary[b];
      point[i] = b;
      i++;
      b = parent_node[b];
      if (b == vocab_size * 2 - 2) break;
    }
    vocab[a].codelen = i;
    vocab[a].point[0] = vocab_size - 2;
    for (b = 0; b < i; b++) {
      vocab[a].code[i - b - 1] = code[b];
      vocab[a].point[i - b] = point[b] - vocab_size;
    }
  }
  free(count);
  free(binary);
  free(parent_node);
}

void LearnVocabFromTrainFile() {
  char word[MAX_STRING];
  FILE *fin;
  long long a, i;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  vocab_size = 0;
  AddWordToVocab((char *)"</s>");
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    train_words++;
    if ((debug_mode > 1) && (train_words % 100000 == 0)) {
      printf("%lldK%c", train_words / 1000, 13);
      fflush(stdout);
    }
    i = SearchVocab(word);
    if (i == -1) {
      a = AddWordToVocab(word);
      vocab[a].cn = 1;
    } else vocab[i].cn++;
    if (vocab_size > vocab_hash_size * 0.7) ReduceVocab();
  }
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
    printf("Finished Learning vocab from File\n");
  }
  file_size = ftell(fin);
  fclose(fin);

}

void SaveVocab() {
  long long i;
  FILE *fo = fopen(save_vocab_file, "wb");
  for (i = 0; i < vocab_size; i++) fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);
  fclose(fo);
}

void ReadVocab() {
  long long a, i = 0;
  char c;
  char word[MAX_STRING];
  FILE *fin = fopen(read_vocab_file, "rb");
  if (fin == NULL) {
    printf("Vocabulary file not found\n");
    exit(1);
  }
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  vocab_size = 0;
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    a = AddWordToVocab(word);
    fscanf(fin, "%lld%c", &vocab[a].cn, &c);
    i++;
  }
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  fseek(fin, 0, SEEK_END);
  file_size = ftell(fin);
  fclose(fin);
  printf("Finished reading vocabulary");
}

void loadNeighbors() 
{
    char *word; int b; 
    long long a; 
    long long index1, index2, it, nnodes, xb, num_walk, walk_length;
    word = (char*)malloc(MAX_STRING* sizeof(char));
    FILE *finit = fopen(neighborFile, "rb");

    fscanf(finit, "%lld %lld %lld", &nnodes, &num_walk, &walk_length);
    max_neighbors = num_walk * walk_length;
    printf("nnodes = %lld  max_neighbors= %lld\n", nnodes, max_neighbors);
    a = posix_memalign((void **)&nbrs, 128, (long long)vocab_size * max_neighbors *sizeof(long long));
    if (nbrs == NULL){printf("Memory allocation failed\n"); exit(1);}
  
    a = posix_memalign((void **)&n2vvocab, 128, (long long)vocab_size * 1 * sizeof(struct n2vword));
    if (n2vvocab==NULL){printf("Memory allocation failed\n");}
  
    
    for (a = 0; a < vocab_size; a++)
    {
      n2vvocab[a].index = a; 
      n2vvocab[a].cn = 0; 

      for (b = 0; b < max_neighbors; b++)
      {
        nbrs[a * max_neighbors + b] = -1;
      }
    }

    for (it=0; it<nnodes; it++)
    {
      fscanf(finit,"%s",word);
      index1 = SearchVocab(word);
      n2vvocab[index1].cn++; 

      if (debug_mode > 3) printf("word = %s, index=%lld \n",word, index1);

      if (index1 < 0) {
        printf("[nbr] Vocabulary does not exist \n");
      }
      xb = 0; 
      for (; xb<max_neighbors; xb++)
      {
          //printf("nbr%lld\n",nbrs[xb + index1*max_neighbors]);
          if (nbrs[xb + index1*max_neighbors]==-1) break ;
      }

      for (b=0; b<walk_length; b++)
      {
        fscanf(finit,"%s",word);
        if (debug_mode > 3) printf("[nbr] word=%s \n",word);

        if (strcmp(word,"-1")==0)
        {
          continue;
        }

        index2 = SearchVocab(word);
        if (index2 < 0) {
          printf("[nbr] Vocabulary does not exist \n");
          continue;
        }

        if (index1 >= 0 && index2 >= 0)
        {
          nbrs[xb + index1*max_neighbors] = index2;
          n2vvocab[index2].cn++;
          xb++; 
        }
        if (debug_mode > 3) {
          printf("[nbr] index1 =%lld, index2=%lld\n",index1, index2);
        }

      }

      if (debug_mode > 3)
      {
        xb = 0; 
        for (; xb<max_neighbors; xb++)
        {
          printf("nbr%lld\n",nbrs[xb + index1*max_neighbors]);
        }
      }

    }

    fclose(finit);
    free(word);
}

void loadLabelFile()
{

  char *word;
  long long a;   
  long long labelval; 
  long long index, it; 
  long long nnodes; 

  word = (char*)malloc(MAX_STRING* sizeof(char));

  a = posix_memalign((void **)&label_sent, 128, (long long)vocab_size * 1 *sizeof(long long));
  if (label_sent == NULL){printf("Memory allocation failed\n"); exit(1);}
  for (a = 0; a < vocab_size; a++)
  {
    label_sent[a] = -1;
  }

  FILE *finit = fopen(labelFile, "rb");
  fscanf(finit, "%lld %d", &nnodes, &nlabels);
  if (debug_mode > 3) printf("nnodes=%lld, nlabels=%d\n", nnodes, nlabels);

  for (it=0; it<nnodes; it++)
  {
    fscanf(finit,"%s",word);
    index = SearchVocab(word);
    fscanf(finit,"%lld",&labelval);
    if (index <= 0)
    {
      if (debug_mode > 3) printf("sentence not on the list");
    }
    else
    {
      if (debug_mode > 3) printf("index=%lld label=%lld\n",index, labelval);
      label_sent[index] = labelval; 
    }  
  }

  free(word); 
  fclose(finit);
}





void InitNet() {
  long long a, b;
  unsigned long long next_random = 1;
  a = posix_memalign((void **)&syn0, 128, (long long)vocab_size * layer1_size * sizeof(real));
  if (syn0 == NULL) {printf("Memory allocation failed\n"); exit(1);}

// Modification for neighbor 
  a = posix_memalign((void **)&temp, 128, (long long)1 * layer1_size * sizeof(real));
  if (temp == NULL) {printf("Memory allocation failed\n"); exit(1);}

  a = posix_memalign((void **)&syn0temp, 128, (long long)1 * layer1_size * sizeof(real));
  if (syn0temp == NULL) {printf("Memory allocation failed\n"); exit(1);}

// Modification for label 
  a = posix_memalign((void **)&templabel, 128, (long long)1 * layer1_size * sizeof(real));
  if (templabel == NULL) {printf("Memory allocation failed\n"); exit(1);}


  if (hs) {
    a = posix_memalign((void **)&syn1, 128, (long long)vocab_size * layer1_size * sizeof(real));
    if (syn1 == NULL) {printf("Memory allocation failed\n"); exit(1);}
    for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++)
     syn1[a * layer1_size + b] = 0;
  }
  if (negative>0) {
    a = posix_memalign((void **)&syn1neg, 128, (long long)vocab_size * layer1_size * sizeof(real));
    if (syn1neg == NULL) {printf("Memory allocation failed\n"); exit(1);}
    for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++)
     syn1neg[a * layer1_size + b] = 0;
  }

  // for labels
  printf("nlabels =%d\n",nlabels);
  a = posix_memalign((void **)&syn1label, 128, (long long)nlabels * layer1_size * sizeof(real));
  if (syn1label == NULL) {printf("Memory allocation failed\n"); exit(1);}
  for (a = 0; a < nlabels; a++) for (b = 0; b < layer1_size; b++)
     syn1label[a * layer1_size + b] = 0;
  
  for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++) {
      next_random = next_random * (unsigned long long)25214903917 + 11;
      syn0[a * layer1_size + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / layer1_size;
  }
  CreateBinaryTree();
  
}



void DestroyNet() {
  if (syn0 != NULL) {
    free(syn0);
  }
  if (syn1 != NULL) {
    free(syn1);
  }
  if (syn1neg != NULL) {
    free(syn1neg);
  }
  if (nbrs != NULL){
    free(nbrs);
  }
  if(syn0temp!=NULL)
  {
    free(syn0temp);
  }
  if(temp!=NULL)
  {
    free(temp);
  }
  if(templabel!=NULL)
  {
    free(templabel);
  }
  if(syn1label!=NULL)
  {
    free(syn1label);
  }
}

void *TrainModelThread(void *id) {
  long long a, b, d, nbr, cw, word, last_word, sentence_length = 0, sentence_position = 0;
  long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
  long long l1, l2,l1n, nbrindex, c, target, label, local_iter = iter;
  unsigned long long next_random = (long long)id;
  int xb, nlab; 
  long long sentence_label;
  real f, g, class_w;
  clock_t now;
  real *neu1 = (real *)calloc(layer1_size, sizeof(real));
  real *neu1e = (real *)calloc(layer1_size, sizeof(real));
  FILE *fi = fopen(train_file, "rb");

  printf("Num threads =%d\n", num_threads);

  fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
  while (1) {
    if (word_count - last_word_count > 10000) {
      word_count_actual += word_count - last_word_count;
      last_word_count = word_count;
      if ((debug_mode > 1)) {
        now=clock();
        printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha,
         word_count_actual / (real)(iter * train_words + 1) * 100,
         word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
        fflush(stdout);
      }
      alpha = starting_alpha * (1 - word_count_actual / (real)(iter * train_words + 1));
      if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
    }
    if (sentence_length == 0) {
      while (1) {
        word = ReadWordIndex(fi);
        if (feof(fi)) break;
        if (word == -1) continue;
        word_count++;
        if (word == 0) break;
        // The subsampling randomly discards frequent words while keeping the ranking same
        if (sample > 0) {
          real ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn;
          next_random = next_random * (unsigned long long)25214903917 + 11;
          if (ran < (next_random & 0xFFFF) / (real)65536) continue;
        }
        sen[sentence_length] = word;
        sentence_length++;
        if (sentence_length >= MAX_SENTENCE_LENGTH) break;
      }
      sentence_position = 0;
      //save the syn0
      for (c=0 ; c<layer1_size; c++) syn0temp[c] = syn0[c+sen[0]*layer1_size];
    }
    if (feof(fi) || (word_count > train_words / num_threads)) {
      word_count_actual += word_count - last_word_count;
      local_iter--;
      if (local_iter == 0) break;
      word_count = 0;
      last_word_count = 0;
      sentence_length = 0;
      fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
      continue;
    }
    word = sen[sentence_position];
    if (word == -1) continue;
    for (c = 0; c < layer1_size; c++) neu1[c] = 0;
    for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
    next_random = next_random * (unsigned long long)25214903917 + 11;
    b = next_random % window;
    if (cbow) {  //train the cbow architecture
      // in -> hidden
      cw = 0;
      for (a = b; a < window * 1 + 1 - b; a++) if (a != window) {
        c = sentence_position - window + a;
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        if (sentence_vectors && (c == 0)) continue;
        last_word = sen[c];
        if (last_word == -1) continue;
        for (c = 0; c < layer1_size; c++) neu1[c] += syn0[c + last_word * layer1_size];
        cw++;
      }
      if (sentence_vectors) {
        last_word = sen[0];
        if (last_word == -1) continue;
        for (c = 0; c < layer1_size; c++) neu1[c] += syn0[c + last_word * layer1_size];
        cw++;
      }
      if (cw) {
        for (c = 0; c < layer1_size; c++) neu1[c] /= cw;
        if (hs) for (d = 0; d < vocab[word].codelen; d++) {
          f = 0;
          l2 = vocab[word].point[d] * layer1_size;
          // Propagate hidden -> output
          for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1[c + l2];
          if (f <= -MAX_EXP) continue;
          else if (f >= MAX_EXP) continue;
          else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
          // 'g' is the gradient multiplied by the learning rate
          g = (1 - vocab[word].code[d] - f) * alpha;
          // Propagate errors output -> hidden
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2];
          // Learn weights hidden -> output
          for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * neu1[c];
        }
        // NEGATIVE SAMPLING
        if (negative > 0) for (d = 0; d < negative + 1; d++) {
          if (d == 0) {
            target = word;
            label = 1;
          } else {
            next_random = next_random * (unsigned long long)25214903917 + 11;
            target = table[(next_random >> 16) % table_size];
            if (target == 0) target = next_random % (vocab_size - 1) + 1;
            if (target == word) continue;
            label = 0;
          }
          l2 = target * layer1_size;
          f = 0;
          for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1neg[c + l2];
          if (f > MAX_EXP) g = (label - 1) * alpha;
          else if (f < -MAX_EXP) g = (label - 0) * alpha;
          else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
          for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * neu1[c];
        }
        // hidden -> in
        for (a = b; a < window * 1 + 1 - b; a++) if (a != window) {
          c = sentence_position - window + a;
          if (c < 0) continue;
          if (c >= sentence_length) continue;
          if (sentence_vectors && (c == 0)) continue;
          last_word = sen[c];
          if (last_word == -1) continue;
          for (c = 0; c < layer1_size; c++) syn0[c + last_word * layer1_size] += neu1e[c];
        }
        if (sentence_vectors) {
          last_word = sen[0];
          if (last_word == -1) continue;
          for (c = 0; c < layer1_size; c++) syn0[c + last_word * layer1_size] += neu1e[c];
        }
      }
    } else {  //train skip-gram
      for (a = b; a < window * 2 + 1 + sentence_vectors - b; a++) if (a != window) {
        c = sentence_position - window + a;
        if (sentence_vectors) if (a >= window * 2 + sentence_vectors - b) c = 0;
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = sen[c];
        if (last_word == -1) continue;
        l1 = last_word * layer1_size;
        for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
        // HIERARCHICAL SOFTMAX
        // Code omitted 
        // NEGATIVE SAMPLING
        if (negative > 0) for (d = 0; d < negative + 1; d++) {
          if (d == 0) {
            target = word;
            label = 1;
          } else {
            next_random = next_random * (unsigned long long)25214903917 + 11;
            target = table[(next_random >> 16) % table_size];
            if (target == 0) target = next_random % (vocab_size - 1) + 1;
            if (target == word) continue;
            label = 0;
          }
          l2 = target * layer1_size;
          f = 0;
          for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1neg[c + l2];
          if (f > MAX_EXP) g = (label - 1) * alpha;
          else if (f < -MAX_EXP) g = (label - 0) * alpha;
          else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
          for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * syn0[c + l1];
        }
        // Learn embedding
        for (c = 0; c < layer1_size; c++) syn0[c + l1] += neu1e[c];
      }
    }
    sentence_position++;
    if (sentence_position >= sentence_length) {
      // Skip objective for node2vec 
      // init 
      l1n = sen[0] * max_neighbors;
      l1  = sen[0] * layer1_size; //src
      xb = 0;
      for ( ; xb<max_neighbors; xb++)
      {
        if (nbrs[xb+l1n]<0)
        {
          break;
        }
      }

      for (c=0 ; c<layer1_size; c++)
      {
        temp[c] = syn0temp[c];
        templabel[c] = syn0temp[c];
      }

    
      for (nbr =  0; nbr< max_neighbors; nbr++)
      {
            //target
            nbrindex = nbrs[l1n+nbr]; 
            if (nbrindex < 0) break; 
            for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
            if (negative > 0) for (d = 0; d < negative + 1; d++) {
              if (d == 0) {
                 target = nbrindex;
                 label = 1;
              } else {
                  next_random = next_random * (unsigned long long)25214903917 + 11;
                  target = table_n2v[(next_random >> 16) % table_size];
                  //if (target == 0) target = next_random % (vocab_size - 1) + 1;
                  if (target == nbrindex) continue;
                  label = 0;
              }
              l2 = target * layer1_size;
              f = 0;
              for (c = 0; c < layer1_size; c++) f += temp[c] * syn1neg[c + l2];
              if (f > MAX_EXP) g = (label - 1) * alpha;
              else if (f < -MAX_EXP) g = (label - 0) * alpha;
              else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
              for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
              for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * temp[c];
          }
         
          if  (debug_mode>3)
          {
            printf("Working for source=%lld and nbr=%lld\n",sen[0], nbrindex);
          }
          for (c = 0; c < layer1_size; c++) temp[c] += neu1e[c];
      }

      // Start working for labels 
      sentence_label  = label_sent[sen[0]];

      if (sentence_label < 0)
      {
        if (xb > 0 )
        {
          for (c=0 ; c<layer1_size; c++)
          {
             syn0[c+l1] = syn0temp[c] + ((beta + beta_label)* (syn0[c+l1]- syn0temp[c]) ) + ((1.0-beta-beta_label)*(temp[c] -syn0temp[c]));
          }
        }
      }else
      {
        if (debug_mode> 3) printf("Working for label %lld\n", sentence_label);
       

        // calculating the loss of a particular instance
        f = 0.0; 
        for (nlab =  0; nlab < nlabels; nlab++)
        {
          class_w = 0.0 ; 
          for(c=0 ; c<layer1_size; c++) class_w = class_w + (templabel[c] * syn1label[c + nlab*layer1_size]);
          f = f + exp(class_w);
        }

         for (c = 0; c < layer1_size; c++) neu1e[c] = 0.0;
        for (nlab = 0; nlab <nlabels; nlab++)
        {
          if (nlab == sentence_label) label = 1.0; 
          else label = 0.0; 
          class_w = 0.0;
          for(c=0 ; c<layer1_size; c++) class_w = class_w + (templabel[c] * syn1label[c + nlab*layer1_size]);

          l2 = nlab * layer1_size; 
          g = (label - (class_w / f)) * alpha; 
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1label[c + l2];
          for (c = 0; c < layer1_size; c++) syn1label[c + l2] += g * templabel[c];
        }

        for (c = 0; c < layer1_size; c++) templabel[c] += neu1e[c];

        if (xb >0)
        {
          for (c=0 ; c<layer1_size; c++)
          {
            syn0[c+l1] = syn0temp[c] + (beta * (syn0[c+l1]- syn0temp[c]) ) + ((beta_label)*(templabel[c] -syn0temp[c]));
            syn0[c+l1] = syn0[c+l1] + ((1.0 - beta - beta_label) * (temp[c] - syn0temp[c]));
          }
        }
        else
        {
          for (c=0 ; c<layer1_size; c++)
          {

            syn0[c+l1] = syn0temp[c] + ((beta + (1.0 - beta - beta_label)) * (syn0[c+l1]- syn0temp[c])) + (beta_label*(templabel[c] - syn0temp[c]));
          }
        }
      }

      sentence_length = 0;
      continue;
    }
  }
  //printf("Closing file\n");
  fclose(fi);
  free(neu1);
  free(neu1e);
  //printf("Exiting\n");
  pthread_exit(NULL);

}

void TrainModel() {
  long a, b, c, d;
  FILE *fo;
  pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
  printf("Starting training using file %s\n", train_file);
  starting_alpha = alpha;
  if (read_vocab_file[0] != 0) ReadVocab(); else LearnVocabFromTrainFile();
  if (save_vocab_file[0] != 0) SaveVocab();

  if (output_file[0] == 0) {printf("Please provide an output file"); return;}
  if (neighborFile[0]!= 0) {
     printf("Loading neighbors\n");
     loadNeighbors();
  }

  if (labelFile[0]!=0)
  {
    printf("Loading labels\n");
    loadLabelFile();
  }

  InitNet();
  if (negative>0)  InitUnigramTable();
  qsort(&n2vvocab[0], vocab_size, sizeof(struct n2vword), VocabCompareN2V);
  InitUnigramTableN2V();

  
  //printf("Finished Initializing Network\n");

  start = clock();
  for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
  for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
  fo = fopen(output_file, "wb");
  if (classes == 0) {
    // Save the word vectors
    fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
    for (a = 0; a < vocab_size; a++) {
      fprintf(fo, "%s ", vocab[a].word);
      if (binary) for (b = 0; b < layer1_size; b++) fwrite(&syn0[a * layer1_size + b], sizeof(real), 1, fo);
      else for (b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", syn0[a * layer1_size + b]);
      fprintf(fo, "\n");
    }
  } else {
    // Run K-means on the word vectors
    int clcn = classes, iter = 10, closeid;
    int *centcn = (int *)malloc(classes * sizeof(int));
    int *cl = (int *)calloc(vocab_size, sizeof(int));
    real closev, x;
    real *cent = (real *)calloc(classes * layer1_size, sizeof(real));
    for (a = 0; a < vocab_size; a++) cl[a] = a % clcn;
    for (a = 0; a < iter; a++) {
      for (b = 0; b < clcn * layer1_size; b++) cent[b] = 0;
      for (b = 0; b < clcn; b++) centcn[b] = 1;
      for (c = 0; c < vocab_size; c++) {
        for (d = 0; d < layer1_size; d++) cent[layer1_size * cl[c] + d] += syn0[c * layer1_size + d];
        centcn[cl[c]]++;
      }
      for (b = 0; b < clcn; b++) {
        closev = 0;
        for (c = 0; c < layer1_size; c++) {
          cent[layer1_size * b + c] /= centcn[b];
          closev += cent[layer1_size * b + c] * cent[layer1_size * b + c];
        }
        closev = sqrt(closev);
        for (c = 0; c < layer1_size; c++) cent[layer1_size * b + c] /= closev;
      }
      for (c = 0; c < vocab_size; c++) {
        closev = -10;
        closeid = 0;
        for (d = 0; d < clcn; d++) {
          x = 0;
          for (b = 0; b < layer1_size; b++) x += cent[layer1_size * d + b] * syn0[c * layer1_size + b];
          if (x > closev) {
            closev = x;
            closeid = d;
          }
        }
        cl[c] = closeid;
      }
    }
    // Save the K-means classes
    for (a = 0; a < vocab_size; a++) fprintf(fo, "%s %d\n", vocab[a].word, cl[a]);
    free(centcn);
    free(cent);
    free(cl);
  }
  fclose(fo);
  free(table);
  free(table_n2v);
  free(pt);
  DestroyVocab();
}

int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
    if (a == argc - 1) {
      printf("Argument missing for %s\n", str);
      exit(1);
    }
    return a;
  }
  return -1;
}

int main(int argc, char **argv) {
  int i;
  if (argc == 1) {
    printf("WORD VECTOR estimation toolkit v 0.1c\n\n");
    printf("Options:\n");
    printf("Parameters for training:\n");
    printf("\t-train <file>\n");
    printf("\t\tUse text data from <file> to train the model\n");
    printf("\t-output <file>\n");
    printf("\t\tUse <file> to save the resulting word vectors / word clusters\n");
    printf("\t-size <int>\n");
    printf("\t\tSet size of word vectors; default is 100\n");
    printf("\t-window <int>\n");
    printf("\t\tSet max skip length between words; default is 5\n");

    printf("\t-neighbor\n");
    printf("\t\tUsed to input neighbor of words/sentences\n");

    printf("\t-label\n");
    printf("\t\tUsed to input label information\n");

    printf("\t-sample <float>\n");
    printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency in the training data\n");
    printf("\t\twill be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)\n");
    printf("\t-hs <int>\n");
    printf("\t\tUse Hierarchical Softmax; default is 0 (not used)\n");
    printf("\t-negative <int>\n");
    printf("\t\tNumber of negative examples; default is 5, common values are 3 - 10 (0 = not used)\n");
    printf("\t-threads <int>\n");
    printf("\t\tUse <int> threads (default 12)\n");
    printf("\t-iter <int>\n");
    printf("\t\tRun more training iterations (default 5)\n");
    printf("\t-min-count <int>\n");
    printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");


    printf("\t-alpha <float>\n");
    printf("\t\tSet the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW\n");

    printf("\t-beta <float>\n");
    printf("\t\tSet relative importance\n");


    printf("\t-classes <int>\n");
    printf("\t\tOutput word classes rather than word vectors; default number of classes is 0 (vectors are written)\n");
    printf("\t-debug <int>\n");
    printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
    printf("\t-binary <int>\n");
    printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
    printf("\t-save-vocab <file>\n");
    printf("\t\tThe vocabulary will be saved to <file>\n");
    printf("\t-read-vocab <file>\n");
    printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
    printf("\t-cbow <int>\n");
    printf("\t\tUse the continuous bag of words model; default is 1 (use 0 for skip-gram model)\n");
    printf("\t-sentence-vectors <int>\n");
    printf("\t\tAssume the first token at the beginning of each line is a sentence ID. This token will be trained\n");
    printf("\t\twith full sentence context instead of just the window. Use 1 to turn on.\n");
    printf("\t\tMaximum 30 * 0.7 = 21M words in the vocabulary. If you want more words to be in the vocabulary please change the hash size\n");
    printf("\nExamples:\n");
    printf("./joint_learner_supervised -train data.txt -output vec.txt  -neighbor neighborfile -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1 -iter 3");
    printf("-sentence-vectors 0 -beta 0.05 \n\n");
    return 0;
  }

  output_file[0] = 0;
  save_vocab_file[0] = 0;
  read_vocab_file[0] = 0;
  if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);

  if ((i = ArgPos((char *)"-save-vocab", argc, argv)) > 0) strcpy(save_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-read-vocab", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-cbow", argc, argv)) > 0) cbow = atoi(argv[i + 1]);

  if (cbow) alpha = 0.05;

  if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-beta", argc, argv)) > 0) beta = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-beta-label", argc, argv)) > 0) beta_label = atof(argv[i + 1]);


  if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-init", argc, argv)) > 0) strcpy(initfromFile, argv[i + 1]);
  if ((i = ArgPos((char *)"-neighbor", argc, argv))>0) strcpy(neighborFile, argv[i+1]);
  if ((i = ArgPos((char *)"-label", argc, argv))>0) strcpy(labelFile, argv[i+1]);


  if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-hs", argc, argv)) > 0) hs = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-classes", argc, argv)) > 0) classes = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-sentence-vectors", argc, argv)) > 0) sentence_vectors = atoi(argv[i + 1]);
  

  vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
  vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
  expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
  for (i = 0; i < EXP_TABLE_SIZE; i++) {
    expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
    expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
  }
  TrainModel();
  DestroyNet();
  free(vocab_hash);
  free(expTable);
  return 0;
}