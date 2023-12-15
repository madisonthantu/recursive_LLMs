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

def main():
    parser = argparse.ArgumentParser(description='running parameter experiments')
    parser.add_argument('--data_dir', type=str, default='initial_datasets')
    parser.add_argument('--dataset_name', type=str, default='')
    args = parser.parse_args()
    # args = vars(parsed_args)
    
    dataset_configs = {
        'news': ["cnn_dailymail", "2.0.0"],
        'reddit': ["reddit_tifu", 'long'],
        'dialogue': ['samsum']
    }
    
    dataset_name = args.dataset_name
    dataset_config = dataset_configs[dataset_name]
    # for dataset_name, dataset_config in dataset_configs.items():
    print(f"Saving {dataset_name} ...")
    data_dict = load_dataset(*dataset_config)
    dataset = concatenate_datasets([data_dict[k] for k in data_dict.keys()])
    for col in dataset.column_names:
        if col in globals.new_col_names.keys():
            dataset = dataset.rename_column(col, globals.new_col_names[col])
    dataset = dataset.select_columns(['document', 'summary'])
    print(dataset)
    dataset = dataset.filter(lambda sample: sample['document'] != '')
    dataset = dataset.filter(lambda sample: sample['summary'] != '')
    
    if args.dataset_name == 'reddit':
        dataset = dataset.shard(num_shards=2, index=0)
    elif args.dataset_name == 'news':
        dataset = dataset.shard(num_shards=8, index=0)
    
    dataset = dataset.add_column(name="id", column=np.arange(dataset.num_rows, dtype=int))
    print(dataset)
    
    path = os.path.join(args.data_dir, dataset_name)
    if not os.path.exists(path):
        os.makedirs(path)
    dataset.to_csv(os.path.join(path, 'full_data.csv'), index=False)
    
    print(data_dict)
        
        
if __name__ == "__main__":
    main()