#!/bin/bash

# Check if CSV folder and PARQUET folder paths are provided as arguments
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Error: Please provide both the path to the CSV folder and the path to the Parquet folder as arguments."
    exit 1
fi

CSV_FOLDER="$1"
PARQUET_FOLDER="$2"

for file in "$CSV_FOLDER"/*.csv.gz; do
    filename=$(basename "$file" .h5.csv.gz)
    python "$(dirname "$0")/convert_to_parquet.py" "$file" "$PARQUET_FOLDER/$filename.parquet"	
done

