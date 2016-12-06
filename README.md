# Discourse Informed Sentence To Vector 
Vector representation of sentences is important for many text processing tasks that involve clustering, 
classifying, or ranking sentences. Recently, distributed representation of sentences learned by neural 
models from unlabeled data has been shown to outperform the traditional bag-of-words representation. 
However, most of these learning methods  consider only the content of a sentence and disregard the 
relations among sentences in a discourse by and large.   
 
In this paper, we propose a series of novel models for learning latent representations of sentences (Sen2Vec) 
that consider the content of a sentence as well as inter-sentence relations. We first represent 
the inter-sentence relations with a language network and then use the network to induce contextual 
information into the content-based Sen2Vec models. Two different approaches are introduced to exploit the information in the network. 
Our first approach retrofits (already trained) Sen2Vec vectors with 
respect to the network in two different ways: (1) using the adjacency relations of a node, 
and (2) using a stochastic sampling method which is more flexible in sampling neighbors of a node. 
The second approach uses a regularizer to encode the information in the network into the existing Sen2Vec model. 
Experimental results show that our proposed models outperform existing methods in 
three fundamental information system tasks demonstrating the effectiveness of our approach. 
The models leverage the computational power of multi-core CPUs to achieve fine-grained computational efficiency. 
We make our code publicly available upon acceptance.

## Requirements
* [Anaconda with Python 3.5](https://www.continuum.io/downloads)
* [ROUGE-1.5.5](http://www.berouge.com/Pages/DownloadROUGE.aspx)

## Python Environment setup and Update

1. Copy the sen2vec_environment.yml file into anaconda/envs folder
2. Get into anaconda/envs folder.
3. Run the following command:

```
conda env create -f sen2vec_environment.yml
```

Now, you have successfully installed sen2vec environment and now you can activate the environment using the following command. 

```
source activate sen2vec
```


If you have added more packages into the environment, you 
can update the .yml file using the following command: 

```
conda env export > sen2vec_environment.yml
```

## ROUGE Environment setup
Please go to the ROUGE directory and run the following command to check whether 
the provided perl script will work or not:
```
./ROUGE-1.5.5.pl 
```

If it shows the options for running the script, then you are fine. However, if it shows 
you haven't have XML::DOM installed then please type following command to install 
it: 

```
cpan XML::DOM
```
Here, CPAN stands for Comprehensive Perl Archive Network. 

## Database Creation and update 

If you have already installed [postgresql](http://postgresapp.com/), then 
you can create a table with the following command for the newsgroup [news] dataset: 

```
psql -c "create database news"
```

After creating the database, use pg_restore to create the schemas which is agnostic to 
the dataset: 

```
pg_restore --jobs=3 --exit-on-error --no-owner --dbname=news sql_dump.dump
```

or 
```
pg_restore --jobs=3 -n public --exit-on-error --no-owner --dbname=news sql_dump.dump
```

We are assuming that either you are using `postgres` as the username or any other username
which already has all the required privileges. To change the password for the `postgres` user,
use the following command-

```
psql -h localhost -d news -U postgres -w
\password
```

If you have made any changes to the database, you can updated the dump 
file using following command (schema only): 

[You may need to set peer authentication: [Peer authentication](http://stackoverflow.com/questions/10430645/how-can-i-get-pg-dump-to-authenticate-properly)]

```
sudo -u postgres pg_dump -s --no-owner -FC news >sql-dump.dump 
```

## Setting Environment Variables

Set the dataset folder path and the connection string in the environment.sh file properly and 
then run the following command-

```
source environment.sh #Unix, os-x
```

## Creating Executable for Word2Vec (Mikolov's Implementation)
Please go to the word2vec code directory inside the project and 
type the following command for creating executable:

```
make clean
make
```

## Installation of Theano for Skip-Thought
```
pip install theano
sudo apt install nvidia-cuda-toolkit
```


## Running the Project 
Run sen2vec with -h argument to see all possible options:

```
python sen2vec -h
usage: sen2vec [-h] -dataset DATASET -ld LD

Sen2Vec

optional arguments:
  -h, --help            show this help message and exit
  -dataset DATASET, --dataset DATASET
                        Please enter dataset to work on [reuter, news]
  -ld LD, --ld LD       Load into Database [0, 1]
  
  -pd PD, --pd PD       Prepare Data [0, 1]
  
  -rbase RBASE, --rbase RBASE       Run the Baselines [0, 1]
  
  -gs GS, --gs GS       Generate Summary [0, 1]
```

For example, you can run for the news dataset using the following command-

```
python sen2vec -dataset news -ld 1 -pd 1 -rbase 1 -gs 1
```

