#!/bin/bash
# Path to Reuters data
#export REUTERS_PATH=~/Dropbox/Journey_to_IUPUI/NLP/data_collection/nlp_data/reuters/reuter21578_temp
export REUTERS_PATH=~/Dropbox/Journey_to_IUPUI/NLP/data_collection/nlp_data/reuters/reuters21578
# export REUTERS_PATH=./sen2vec/documentReader/reuters21578
# database, username, passwd, host, port
export REUTERS_DBSTRING=reuter,postgres,postgres,localhost,5432
# export REUTERS_DBSTRING=reuters,naeemul,naeemul,localhost,5432
export NEWSGROUP_DBSTRING=news,postgres,postgres,localhost,5432
export NEWSGROUP_PATH=~/Dropbox/Journey_to_IUPUI/NLP/data_collection/nlp_data/newsgroup/20news-bydate



export P2VECSENTRUNNERINFILE=~/Documents/sen2vec/Data/sents
export P2VECSENTRUNNEROUTFILE=~/Documents/sen2vec/Data/sents_repr
export TRTESTFOLDER=~/Documents/sen2vec/Data/

export N2VOUTFILE=~/Documents/sen2vec/Data/node_repr
export GINTERTHR=0.8
export GINTRATHR=0.5
export GTHRSUM=0.1
export DUMPFACTOR=0.85
export TOPNSUMMARY=0.2
export ITERUPDATESEN2VECFILE=~/Documents/sen2vec/Data/retrofitted_repr
export GRAPHFILE=~/Documents/sen2vec/Data/graph_0.8_0.5

