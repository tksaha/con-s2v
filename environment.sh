#!/bin/bash
# Path to Reuters data
export REUTERS_PATH=/Users/tksaha/Dropbox/Journey_to_IUPUI/NLP/data_collection/nlp_data/reuters/reuters21578
# export REUTERS_PATH=./sen2vec/documentReader/reuters21578
# database, username, passwd, host, port
export REUTERS_DBSTRING=news,postgres,postgres,localhost,5432
# export REUTERS_DBSTRING=reuters,naeemul,naeemul,localhost,5432
export P2VECSENTRUNNERINFILE=/Users/tksaha/Dropbox/Journey_to_IUPUI/NLP/gitlabcodes/sen2vec/Data/sents
export P2VECSENTRUNNEROUTFILE=/Users/tksaha/Dropbox/Journey_to_IUPUI/NLP/gitlabcodes/sen2vec/Data/sents_repr

export N2VOUTFILE=/Users/tksaha/Dropbox/Journey_to_IUPUI/NLP/gitlabcodes/sen2vec/Data/node_repr
export GINTERTHR=0.4
export GINTRATHR=0.1

