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

from datasets import load_dataset

def main():
    parser = argparse.ArgumentParser(description='running parameter experiments')
    parser.add_argument('--data_dir', type=str, default='synthetic_datasets')
    parser.add_argument('--file_name', type=str, default='full_data.csv')
    parser.add_argument('--dataset_name', type=str, default='')
    parser.add_argument('--output_dir', type=str, default='synthetic_datasets')
    args = parser.parse_args()
    
    dataset = load_dataset(args.data_dir, data_files=args.file_name)['train']
    print(dataset)
    dataset = dataset.filter(lambda sample: sample['document'] != '')
    dataset = dataset.filter(lambda sample: sample['summary'] != '')
    print(dataset)
    
    data_dict = dataset.train_test_split(test_size=0.3)
    path = os.path.join(args.output_dir, args.dataset_name)
    if not os.path.exists(path):
        os.makedirs(path)
    print(data_dict)
    
    data_dict['train'].to_csv(os.path.join(path, 'training_data.csv'), index=False)
    data_dict['test'].to_csv(os.path.join(path, 'validation_data.csv'), index=False)
    
if __name__ == "__main__":
    main()