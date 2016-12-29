#!/bin/bash
export SEN2VEC_DIR=~/Documents/news/sen2vec


export NEWSGROUP_PATH=$SEN2VEC_DIR/Data/newsgroup/
export NEWSGROUP_DBSTRING=news,postgres,postgres,localhost,5432


export DOC2VECEXECDIR=$SEN2VEC_DIR/sen2vec/word2vec/word2vec
export RETROFITONEEXE=$SEN2VEC_DIR/sen2vec/word2vec/retrofit_word2vec_one
export REGSEN2VECEXE=$SEN2VEC_DIR/sen2vec/word2vec/reg_sen2vec_net
export JOINTSUPEXE=$SEN2VEC_DIR/sen2vec/word2vec/joint_learner


export TRTESTFOLDER=$SEN2VEC_DIR/Data
export GINTERTHR=0.8
export GINTRATHR=0.5

export GTHRSUMTFIDF=0.1
export GTHRSUMLAT=0.1
export DUMPFACTOR=0.85


# TOPNSUMMARY should be set to 0.2 for classification tasks (20%) and to 1.0 for ranking tasks
export TOPNSUMMARY=0.2
export KNEIGHBOR=20



export ROUGE=$SEN2VEC_DIR/sen2vec/rouge/ROUGE-1.5.5.pl
export ROUGE_EVAL_HOME=$SEN2VEC_DIR/sen2vec/rouge/data
export SUMMARYFOLDER=$SEN2VEC_DIR/Data/Summary/
export MODELSUMMARYFOLDER=$SEN2VEC_DIR/Data/Summary/model
export SYSTEMSUMMARYFOLDER=$SEN2VEC_DIR/Data/Summary/system



export DICTDIR=$SEN2VEC_DIR/Data/lexicons
export DICTREGDICT=wordnet-synonyms.txt



# Theano Stuffs : device=gpu0,lib.cnmem=1' 
export THEANO_FLAGS='floatX=float32'
