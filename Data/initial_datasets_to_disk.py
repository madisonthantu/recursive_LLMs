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
    # print(data_dict)
    dataset = concatenate_datasets([data_dict[k] for k in data_dict.keys()])
    # print(dataset)
    for col in dataset.column_names:
        if col in globals.new_col_names.keys():
            dataset = dataset.rename_column(col, globals.new_col_names[col])
    # print(dataset)
    dataset = dataset.select_columns(['document', 'summary'])
    print(dataset)
    dataset = dataset.add_column(name="id", column=np.arange(dataset.num_rows, dtype=int))
    dataset = dataset.filter(lambda sample: sample['document'] != '')
    dataset = dataset.filter(lambda sample: sample['summary'] != '')
    print(dataset)
    x = dataset.filter(lambda sample: sample['id'] == 6054)
    data_dict = dataset.train_test_split(test_size=0.3)
    path = os.path.join(args.data_dir, dataset_name)
    if not os.path.exists(path):
        os.makedirs(path)
    data_dict['train'].to_csv(os.path.join(path, 'training_data.csv'), index=False)
    data_dict['test'].to_csv(os.path.join(path, 'validation_data.csv'), index=False)
    full_dataset = concatenate_datasets([data_dict[k] for k in data_dict.keys()])
    full_dataset.to_csv(os.path.join(path, 'full_data.csv'), index=False)
        
        
if __name__ == "__main__":
    main()