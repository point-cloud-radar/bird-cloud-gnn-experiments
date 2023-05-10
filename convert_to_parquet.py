#!/usr/bin/env python

# Example of bash usage:
# for file in CSV_FOLDER/*.csv.gz
# do
#   filename=$(basename $file .h5.csv.gz)
#   ../convert_to_parquet.py $file PARQUET_FOLDER/$filename.parquet
# done

import pandas as pd
import os
import sys

def usage():
    print(f"""
    {sys.argv[0]} INPUT_FILE OUTPUT_FILE
    """)

if len(sys.argv) < 3:
    usage()
    exit(1)

in_file = sys.argv[1]
out_file = sys.argv[2]

if not os.path.exists(in_file):
    usage()
    print(f"ERROR: {in_file} does not exist")
    exit(1)
elif os.path.exists(out_file):
    usage()
    print(f"ERROR: {out_file} already exists")
    exit(1)

data = pd.read_csv(in_file)
data.to_parquet(out_file)
