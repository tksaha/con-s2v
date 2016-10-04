# Discourse Informed Sentence To Vector 



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
make
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

