import os
import sys

def main():
    flow = os.getenv('META_FLOW', 'default_flow.py')  # Get the flow name from an environment variable
    os.system(f"python {flow} run")

if __name__ == '__main__':
    main()
