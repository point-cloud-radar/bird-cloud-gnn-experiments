#!/bin/bash

function info_msg {
  echo -e "\033[0;32m$1\033[0m"
}

function error_msg {
  echo -e "\033[0;31m$1\033[0m"
}

function usage {
  info_msg "Usage:

    bash $0 DATAFOLDER

This will iterate over all files in DATAFOLDER and create tables for each of them in the DB 'bird'.
The names of the tables will be the names of the files.
Additionally, it will create a table 'filenames' with the field 'name' saving the name of the file and tables.
"
}

if [ -z "$1" ]; then
  usage
  exit 1
fi

DATA_FOLDER=$1

if [ ! -d "$DATA_FOLDER" ]; then
  error_msg "ERROR: '$DATA_FOLDER' is not a folder"
  usage
  exit 1
fi

psql -d bird -f filenames_table.sql

for file in "$DATA_FOLDER"/*
do
  if [ -f "$file" ] && [[ "$file" == *.csv.gz ]]; then
    info_msg "Processing file $file"

    name=$(basename $file)
    name="${name%%.*}"
    name=$(echo "$name" | tr '[:upper:]' '[:lower:]')

    info_msg "Creating new table $name"
    sed "s/{TABLENAME}/$name/g" create.sql.template > /tmp/tmp.sql
    psql -d bird -f /tmp/tmp.sql

    info_msg "Convert $file to table $name"
    if ! ogr2ogr -append -f CSV -nln "$name" -lco GEOMETRY_NAME=geom -oo EMPTY_STRING_AS_NULL=YES -oo X_POSSIBLE_NAMES=x -oo Y_POSSIBLE_NAMES=y -oo Z_POSSIBLE_NAMES=z Pg:"dbname=bird host=localhost user=postgres password=admin port=5432" "/vsigzip//$file";
    then
      error_msg "ERROR: Something went wrong"
      exit 1
    fi

    info_msg "Adding $name to table 'filenames'"
    psql -d bird -c "INSERT INTO filenames(tablename) VALUES ('$name');"
  fi
done
rm -f tmp.sql
