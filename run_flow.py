import os
import pandas as pd

def main():
    #read data file and save it locally
    #bucket_path = os.getenv('BUCKET_PATH')
    #temporary_data_file = os.getenv('INPUT_DATA_PATH')
    #df = pd.read_csv(bucket_path)
    #df.to_csv(temporary_data_file, index = False)
    #make directory for artifact registry locally
    artifact_dir = os.getenv('ARTIFACT_REGISTRY')
    os.makedirs(artifact_dir, exist_ok = True)
    flow = os.getenv('META_FLOW', 'default_flow.py')  # Get the flow name from an environment variable
    os.system(f"python {flow} run")

if __name__ == '__main__':
    main()
