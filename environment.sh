#!/bin/bash
# Path to Reuters data
#export REUTERS_PATH=~/Documents/sen2vec/Data/reuter21578_temp
export REUTERS_PATH=~/Dropbox/Journey_to_IUPUI/NLP/data_collection/nlp_data/reuters/reuters21578



# export REUTERS_PATH=./sen2vec/documentReader/reuters21578
# database, username, passwd, host, port
export REUTERS_DBSTRING=reuter,postgres,postgres,localhost,5432
# export REUTERS_DBSTRING=reuters,naeemul,naeemul,localhost,5432
export NEWSGROUP_DBSTRING=news,postgres,postgres,localhost,5432
export IMDB_DBSTRING=imdb,postgres,postgres,localhost,5432
export SENTTREE_DBSTRING=senttree,postgres,postgres,localhost,5432

export NEWSGROUP_PATH=~/Dropbox/Journey_to_IUPUI/NLP/data_collection/nlp_data/newsgroup/20news-bydate
export IMDB_PATH=~/Documents/sen2vec/Data/aclImdb
export SENTTREE_PATH=~/Documents/sen2vec/Data/stanfordSentimentTreebank


export P2VECSENTRUNNERINFILE=~/Documents/sen2vec/Data/sents
export P2VECSENTRUNNEROUTFILE=~/Documents/sen2vec/Data/sents_repr

export P2VECRUNNERINFILE=~/Documents/sen2vec/Data/docs
export P2VECRUNNEROUTFILE=~/Documents/sen2vec/Data/docs_repr

export P2VDOCOUT=~/Documents/sen2vec/Data/docs_repr_CEXE
export P2VECRUNNERCEXEOUTFILE=~/Documents/sen2vec/Data/docs_repr_CEXE
export DOC2VECEXECDIR=~/Documents/sen2vec/sen2vec/word2vec/word2vec
export RETROFITONEEXE=~/Documents/sen2vec/sen2vec/word2vec/retrofit_word2vec_one

export P2VCEXECSENTFILE=~/Documents/sen2vec/Data/sentsCEXE
export P2VCEXECOUTFILE=~/Documents/sen2vec/Data/sentsCEXE_repr
export P2VECSENTDOC2VECOUT=~/Documents/sen2vec/Data/sentCEXE_repr



export TRTESTFOLDER=~/Documents/sen2vec/Data

export N2VOUTFILE=~/Documents/sen2vec/Data/node_repr
export GINTERTHR=0.8
export GINTRATHR=0.5

export GTHRSUM=0.1
export DUMPFACTOR=0.85
export TOPNSUMMARY=0.2
export KNEIGHBOR=20

export ITERUPDATESEN2VECFILE=~/Documents/sen2vec/Data/retrofitted_repr
export GRAPHFILE=~/Documents/sen2vec/Data/graph_0.8_0.5

export ROUGE=~/rouge-1.5.5/ROUGE-1.5.5.pl
export ROUGE_EVAL_HOME=~/rouge-1.5.5/data
export SUMMARYFOLDER=~/Research/sen2vec/Data/Summary/
export MODELSUMMARYFOLDER=~/Research/sen2vec/Data/Summary/model
export SYSTEMSUMMARYFOLDER=~/Research/sen2vec/Data/Summary/system
