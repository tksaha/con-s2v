## Requirements
* [Anaconda with Python 3.5](https://www.continuum.io/downloads)

## Installation
1. Copy the sen2vec_environment.yml file into anaconda/envs folder
2. Get into anaconda/envs folder.
3. Run the following command: 

```
conda env create -f sen2vec_environment.yml
```

Now, you have successfully installed sen2vec environment.

## Database Creation 
If you have already installed [postgresql] (http://postgresapp.com/), then 
you can create a table with the following command: 

```
psql -c create database news
```
After creating database use pg_restore to create the schemas agnostic to 
the dataset: 

```
pg_restore --jobs=3 --exit-on-error --no-owner --dbname=news sql_dump.dump
```


```
psql -h localhost -d news -U postgres -w
psql -d news -h localhost -d 5432 -U postgres -w
```

## Run 
Run sen2vec with -h option to see all possible options:

```
python sen2vec -h
```

