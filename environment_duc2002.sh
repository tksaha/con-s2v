#!/bin/bash
# Path to Reuters data
#export REUTERS_PATH=~/Documents/sen2vec/Data/reuter21578_temp
export SEN2VEC_DIR=~/Documents/DUC2002/sen2vec
export REUTERS_PATH=$SEN2VEC_DIR/Data/reuters21578
export SICK_PATH=$SEN2VEC_DIR/Data/sick_data



# export REUTERS_PATH=./sen2vec/documentReader/reuters21578
# database, username, passwd, host, port
export REUTERS_DBSTRING=reuter,postgres,postgres,localhost,5432
# export REUTERS_DBSTRING=reuters,naeemul,naeemul,localhost,5432
export NEWSGROUP_DBSTRING=news,postgres,postgres,localhost,5432
export IMDB_DBSTRING=imdb,postgres,postgres,localhost,5432
export SENTTREE_DBSTRING=senttree,postgres,postgres,localhost,5432
export SENTTREE2WAY_DBSTRING=senttree2way,postgres,postgres,localhost,5432
export SICK_DBSTRING=sick,postgres,postgres,localhost,5432

export NEWSGROUP_PATH=$SEN2VEC_DIR/Data/newsgroup/
export IMDB_PATH=$SEN2VEC_DIR/Data/aclImdb
export SENTTREE_PATH=$SEN2VEC_DIR/Data/stanfordSentimentTreebank



export P2VECSENTRUNNERINFILE=$SEN2VEC_DIR/Data/sents
export P2VECSENTRUNNEROUTFILE=$SEN2VEC_DIR/Data/sents_repr

export P2VECRUNNERINFILE=$SEN2VEC_DIR/Data/docs
export P2VECRUNNEROUTFILE=$SEN2VEC_DIR/Data/docs_repr

export P2VDOCOUT=$SEN2VEC_DIR/Data/docs_repr_CEXE
export P2VECRUNNERCEXEOUTFILE=$SEN2VEC_DIR/Data/docs_repr_CEXE
export DOC2VECEXECDIR=$SEN2VEC_DIR/sen2vec/word2vec/word2vec
export RETROFITONEEXE=$SEN2VEC_DIR/sen2vec/word2vec/retrofit_word2vec_one

export P2VCEXECSENTFILE=$SEN2VEC_DIR/Data/sentsCEXE
export P2VCEXECOUTFILE=$SEN2VEC_DIR/Data/sentsCEXE_repr
export P2VECSENTDOC2VECOUT=$SEN2VEC_DIR/Data/sentCEXE_repr



export TRTESTFOLDER=$SEN2VEC_DIR/Data

export N2VOUTFILE=$SEN2VEC_DIR/Data/node_repr
export GINTERTHR=0.4
export GINTRATHR=0.4

export GTHRSUMTFIDF=0.1
export GTHRSUMLAT=0.1
export DUMPFACTOR=0.85

# TOPNSUMMARY should be set to 0.2 for classification tasks (20%) and to 1.0 for ranking tasks
export TOPNSUMMARY=1.0
export KNEIGHBOR=20


export ITERUPDATESEN2VECFILE=$SEN2VEC_DIR/Data/retrofitted_repr
export GRAPHFILE=$SEN2VEC_DIR/Data/graph_0.4

export ROUGE=$SEN2VEC_DIR/sen2vec/rouge/ROUGE-1.5.5.pl
export ROUGE_EVAL_HOME=$SEN2VEC_DIR/sen2vec/rouge/data
export SUMMARYFOLDER=$SEN2VEC_DIR/Data/Summary/
export MODELSUMMARYFOLDER=$SEN2VEC_DIR/Data/Summary/model
export SYSTEMSUMMARYFOLDER=$SEN2VEC_DIR/Data/Summary/system

export DUC_DBSTRING=duc2002,postgres,postgres,localhost,5432
export DUC_PATH=$SEN2VEC_DIR/Data/DUC_merged


export DUC_LAMBDA=1.0
export DUC_DIVERSITY=0
export DUC_TOPIC=2002
#export DUC_EVAL='TEST'

export REGSEN2VECREPRFILE=$SEN2VEC_DIR/Data/reg_sent
export REGSEN2VECEXE=$SEN2VEC_DIR/sen2vec/word2vec/reg_sen2vec_net

export DICTREGSEN2VECREPRFILE=$SEN2VEC_DIR/Data/dictreg_sent
export DICTDIR=$SEN2VEC_DIR/Data/lexicons

export REG_BETA_UNW=1.0
export REG_BETA_W=1.0
export ITERUPDATE_ALPHA=1.0
export N2VBETA=1.0
export DICTREGDICT=wordnet-synonyms.txt

export JOINTS2VRPRFILE=$SEN2VEC_DIR/Data/joint_sent
export JOINT_BETA=1.0
export NUM_WALKS=10
export WALK_LENGTH=5
export JOINTLEXE=$SEN2VEC_DIR/sen2vec/word2vec/joint_word_node2vec

export FASTS2VRPRFILE=$SEN2VEC_DIR/Data/fast_sent
export FSENT_BETA=0.9

export SJOINTS2VRPRFILE=$SEN2VEC_DIR/Data/joint_sup_sent
export JOINT_SENT_BETA=0.90
export JOINT_SENT_LBETA=0.07
export JOINTSUPEXE=$SEN2VEC_DIR/sen2vec/word2vec/joint_learner

export DBOW_ONLY=1
export NBR_TYPE=0
export LAMBDA=0.8
export FULL_DATA=1


# Theano Stuffs : device=gpu0,lib.cnmem=1' 
export THEANO_FLAGS='floatX=float32'
