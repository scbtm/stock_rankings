import os
import sys
import polars as pl

def main():
    #read data file and save it locally
    bucket_path = os.getenv('BUCKET_PATH')
    temporary_data_file = os.getenv('INPUT_DATA_PATH')
    df = pl.read_csv(bucket_path)
    df.to_csv(temporary_data_file)
    flow = os.getenv('META_FLOW', 'default_flow.py')  # Get the flow name from an environment variable
    os.system(f"python {flow} run")

if __name__ == '__main__':
    main()
