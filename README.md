# bird-cloud-gnn-experiments

## Running postGIS

```bash
docker-compose up
```

Log into the `postgis` image using `docker exec`, e.g.:

```bash
docker exec -it bird-cloud-gnn-experiments-postgis-1 /bin/bash
```

Change to `postgres` user and run `psql`:

```bash
su - postgres
psql
```

Create database, connect and create postgis extension:

```sql
CREATE DATABASE bird;
\c bird;
CREATE EXTENSION postgis;
```

Leave `psql`, and run the script `create_tables.sh` passing the folder with data.
The data folder must contains files ending in `.csv.gz`.

```bash
bash /sql/create_tables.sh /data/DATA_FOLDER/
```

After a while - which can be a long time - tables will be created inside the `bird` DB for each file, and a table called `filenames` containing the table names as well.

### Extra

To access the pgamind, go to <http://localhost:5050>.

To convert a .csv to the postgres format:

```bash
ogr2ogr -overwrite -lco GEOMETRY_NAME=geom -oo EMPTY_STRING_AS_NULL=YES -oo X_POSSIBLE_NAMES=x -oo Y_POSSIBLE_NAMES=y -oo Z_POSSIBLE_NAMES=z Pg:"dbname=bird host=localhost user=postgres password=admin port=5432" small.csv
```

The `ogr2ogr` program is installed in the Docker image `kartoza/postgis`, and you have to `su - postgres` for some reason.
