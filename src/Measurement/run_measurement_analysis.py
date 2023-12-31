import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd

from datasets import load_dataset_builder, load_dataset, concatenate_datasets
import platform
from nltk.probability import FreqDist

import sys
import re
import time
from ratelimiter import RateLimiter
from itertools import chain
from tqdm import tqdm
import os
import argparse
from datasets import load_dataset


# sys.path.append(os.path.abspath("/home/madisonthantu/recursive_LLMs/Data"))
# sys.path.append(os.path.abspath("/Users/madisonthantu/Desktop/COMS_6998/Final_Project/recursive_LLMs/src"))
print(platform.platform())

sys.path.insert(1, '/Users/madisonthantu/Desktop/COMS_6998/Final_Project/recursive_LLMs/src')
# import globals as globals
from Measurement import globals
from Measurement.measurement import Measurement
globals.init()
from utils import *
init()

from googleapiclient.errors import HttpError
from googleapiclient import discovery
import requests

sys.path.insert(1, '../Data')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='running measurement analysis on dataset')
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--dataset_name', type=str, default='')
    parser.add_argument('--generation', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--overwrite_output_dir', type=bool, default=0)
    parser.add_argument('--DEBUG', type=bool, default=0)
    args = parser.parse_args()
    output_measurements_path = args.output_dir
    if os.path.isfile(os.path.join(output_measurements_path, 'dataset_measurements.pkl')):
        if args.overwrite_output_dir:
            print("The dataset measurement file already exists. Do you wish to overwrite it? [yes/no]")
            input1 = input()
            if input1 != 'yes':
                sys.exit()
        else:
            print("To overwrite output directory, set option [--overwrite_output_dir 1]")
            sys.exit()
    assert(args.dataset_name in args.data_path)
    assert(args.dataset_name in args.output_dir)
    if args.generation == 'baseline':
        assert('initial_datasets' in args.data_path)
        assert('baseline' in args.output_dir)
        assert(args.model == 'baseline')
    else:
        assert('synthetic_datasets' in args.data_path)
        assert(args.model in args.data_path)
        assert(args.model in args.output_dir)
        assert(args.model != 'baseline')
        assert(args.dataset_name in args.output_dir)
        assert(f"gen{args.generation}" in args.data_path)
        assert(f"gen{args.generation}" in args.output_dir)
    print(output_measurements_path)
    

    data_df = pd.read_csv(os.path.join(args.data_path, 'full_data.csv'))#, index_col='id')
    # print(data_df.shape)
    data_df = data_df.dropna()
    # print(data_df.shape)
    data_df.document=data_df.document.astype(str)
    data_df.summary=data_df.summary.astype(str)
    dataset_specs = {
        'generation':args.generation, 
        'subject':args.dataset_name,
        'model':args.model,
        'no_total_samples':data_df.shape[0],
    }
    if args.model == 'baseline':
        dataset_specs['no_train_samples'] = pd.read_csv(os.path.join(args.data_path, 'training_data.csv')).shape[0]
        dataset_specs['no_validation_samples'] = pd.read_csv(os.path.join(args.data_path, 'validation_data.csv')).shape[0]
        print("No. training samples =", dataset_specs['no_train_samples'])
        print("No. validation samples =", dataset_specs['no_validation_samples'])
    else:
        f = open(os.path.join(args.data_path, 'config.json'))
        data_config = json.load(f)
        with open(os.path.join(args.output_dir, "dataset_config.json"), "w") as outfile:
            json.dump(data_config, outfile)
    
    # if os.path.exists(args.data_path, 'training_data.csv'):
    #     dataset_specs['no_training_samples'] = pd.read_csv(os.path.join(args.data_path, 'training_data.csv')).shape[0],
    # if os.path.exists(args.data_path, 'validation_data.csv'):
    #     dataset_specs['no_training_samples'] = pd.read_csv(os.path.join(args.data_path, 'validation_data.csv')).shape[0],
        
    print(dataset_specs)
    measurements = Measurement(data_df, dataset_specs, DEBUG=args.DEBUG)
    output_measurements_path = args.output_dir
        
    results = measurements.measure()
    results.update(dataset_specs)
    if not os.path.exists(output_measurements_path):
        os.makedirs(output_measurements_path)
    save_files_to_pkl({
        os.path.join(
            output_measurements_path, 'dataset_measurements.pkl'
        ):results
    })
    