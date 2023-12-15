import os
import sys
import argparse
from tqdm import tqdm
import pickle
import numpy as np
import logging
import pandas as pd
import textwrap
import json 

# sys.path.append(os.path.abspath("/Users/madisonthantu/Desktop/COMS 6998/Final Project/recursive_LLMs/src"))
# sys.path.append(os.path.abspath("/home/madisonthantu/recursive_LLMs/src/Measurement"))
sys.path.insert(1, '/Users/madisonthantu/Desktop/COMS_6998/Final_Project/recursive_LLMs/src')
import Measurement.globals as globals
globals.init()
print(globals.API_URL)

from datasets import concatenate_datasets, load_dataset, DatasetDict, Dataset
from huggingface_hub import login


def init():
    login(token=globals.hug_token)
    

def check_valid_path(file_path, file_name=None):
    if not os.path.isdir(file_path):
        raise Exception(f"Invalid file path supplied <{file_path}>")
    if file_name != None:
        if not os.path.isfile(os.path.join(file_path, file_name)):
            raise Exception(f"Invalid file path supplied <{os.path.join(file_path, file_name)}>")
    

def save_dict_to_json(file_path, file_name, my_dict):
    check_valid_path(file_path)
    for k, v in my_dict.items():
        if isinstance(v, pd.Series):
            my_dict[k] = v.tolist()
    with open(os.path.join(file_path, file_name), "w") as outfile: 
        json.dump(my_dict, outfile)
    
    
# def load_pickle(file_path, file_name):
#     # check_valid_path(file_path, file_name)
#     with open(os.path.join(file_path , file_name), 'rb') as f:
#         return pickle.load(f)
    
    
def load_synthetic_dataset(synthetic_data_path, file_name):
    check_valid_path(synthetic_data_path, file_name)
    data_df = pd.read_pickle(os.path.join(synthetic_data_path, file_name))#.sort_values(by=['id'])
    dataset = Dataset.from_pandas(data_df)
    return dataset


def save_files_to_pkl(path_dict):
    for path, values in path_dict.items():
        pickle.dump(values, open(path, "wb" ) )
    print("Files saved 💃🕺")
    

class ListDataset(Dataset):
     def __init__(self, original_list):
        self.original_list = original_list
     def __len__(self):
        return len(self.original_list)

     def __getitem__(self, i):
        return self.original_list[i]
    
    
def prettyprint(text, val=150):
    print(textwrap.fill(text, val))
    

def prettyprint_dict(d, indent=0, skip_list=['summary_token_distribution', 'samples', 'emotion_intensity_measurements']):
    for key, value in d.items():
        print('\t' * indent + str(key))
        if key in skip_list:
                continue
        elif isinstance(value, dict):
            prettyprint_dict(value, indent+1)
    
    
def make_output_base_path_str(args):
    path = os.path.join(args['output_dir'], args['base_model'], args['dataset'])
    print(f"Output directory path: {path}")
    if os.path.isdir(path):
        print(f"The supplied output directory, {[path]}, already exists. Do you wish to overwrite this directory's contents? [y/n]: ")
        if str(input()).lower() != "y":
            sys.exit()
    return path
    
    
def load_huggingface_dataset(data_config, args):
    data_path = os.path.join(args['data_dir'], globals.dataset_files[args['dataset']])
    if os.path.isfile(data_path):
        print("Reading initial dataset from disk ...")
        data_df = pd.read_pickle(data_path)
        return Dataset.from_pandas(data_df)
    else:
        print("Reading initial dataset from Huggingface ...")
        data_dict = load_dataset(*data_config)
        dataset = concatenate_datasets([data_dict[k] for k in data_dict.keys()])
        for col in dataset.column_names:
            if col in globals.new_col_names.keys():
                dataset = dataset.rename_column(col, globals.new_col_names[col])
        if args['dataset'] == 'reddit':
            dataset = dataset.add_column(name="id", column=np.arange(dataset.num_rows))
        dataset = dataset.filter(lambda sample: len(sample['document'].split()) <= globals.max_src_length[args['dataset']])
        dataset = dataset.filter(lambda sample: len(sample['summary'].split()) <= globals.max_target_length[args['dataset']])
        data_df = dataset.to_pandas()
        save_files_to_pkl({data_path: data_df})
        print("Saved initial dataset to disk ...")
        return dataset.select_columns(['id', 'document', 'summary'])
    
    
def train_val_test_split(dataset):
    data_dict = dataset.train_test_split(test_size=0.3)
    dev_data_dict = data_dict['test'].train_test_split(test_size=0.5)
    return DatasetDict({
        'train': data_dict['train'],
        'validation': dev_data_dict['train'],
        'test': dev_data_dict['test']
    })
    
    
def train_test_split(df):
    dataset = Dataset.from_pandas(df)
    data_dict = dataset.train_test_split(test_size=0.3)
    return data_dict