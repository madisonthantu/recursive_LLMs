import sys
import argparse
import os
import pandas as pd
from datasets import concatenate_datasets, load_dataset, DatasetDict, Dataset
sys.path.append(os.path.abspath("/Users/madisonthantu/Desktop/COMS_6998/Final_Project/recursive_LLMs"))
sys.path.append(os.path.abspath("/home/madisonthantu/recursive_LLMs"))
import src.Measurement.globals as globals
globals.init()
import numpy as np
import json

from datasets import load_dataset

# MAX_LEN = 1024

def main():
    parser = argparse.ArgumentParser(description='running dataset preprocessing')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--file_name', type=str, default='full_data.csv')
    parser.add_argument('--dataset_name', type=str, default='')
    parser.add_argument('--base_model', type=str)
    args = parser.parse_args()
    
    dataset = load_dataset(args.data_dir, data_files=args.file_name)['train']
    print(dataset)
    dataset = dataset.filter(lambda sample: sample['document'] != "" and sample['document'] is not None)
    dataset = dataset.filter(lambda sample: sample['summary'] != "" and sample['summary'] is not None)
    print(dataset)
    
    # dataset = dataset.filter(lambda example: len(example['document']) < MAX_LEN)
    # print(dataset)
    
    data_dict = dataset.train_test_split(test_size=0.3)
    assert(args.dataset_name in args.data_dir), "Data directory and dataset name must match."
    
    path = args.data_dir
    dataset.to_csv(os.path.join(path, 'full_data.csv'), index=False)
    f = open(os.path.join(path, 'config.json'))
    data_config = json.load(f)
    data_config['num_samples'] = len(dataset)
    with open(os.path.join(path, "config.json"), "w") as outfile:
        json.dump(data_config, outfile)
    print(data_config)
        
    data_dict['train'].to_csv(os.path.join(path, 'training_data.csv'), index=False)
    data_dict['test'].to_csv(os.path.join(path, 'validation_data.csv'), index=False)
    
if __name__ == "__main__":
    main()