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

sys.path.insert(1, '../Data')
sys.path.append(os.path.abspath("/home/madisonthantu/recursive_LLMs/Data"))
sys.path.append(os.path.abspath("/Users/madisonthantu/Desktop/COMS_6998/Final_Project/recursive_LLMs/src"))
print(platform.platform())

import globals as globals
from measurement import Measurement
globals.init()
from utils import *

from googleapiclient.errors import HttpError
from googleapiclient import discovery
import requests

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='running measurement analysis on dataset')
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--dataset_name', type=str, default='')
    parser.add_argument('--generation', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--DEBUG', type=bool, default=0)
    
    args = parser.parse_args()
    # print(args.DEBUG)
    output_measurements_path = args.output_dir
    # assert(not os.path.exists(output_measurements_path))
    print(output_measurements_path)
    # sys.exit()

    data_df = pd.read_csv(os.path.join(args.data_path, 'full_data.csv'))
    data_df.document=data_df.document.astype(str)
    data_df.summary=data_df.summary.astype(str)
    dataset_specs = {
        'generation':args.generation, 
        'subject':args.dataset_name,
        'model':args.model,
        'no_total_samples':data_df.shape[0],
        'no_training_samples':pd.read_csv(os.path.join(args.data_path, 'training_data.csv')).shape[0],
        'no_validation_samples':pd.read_csv(os.path.join(args.data_path, 'validation_data.csv')).shape[0]
    }
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

# small_df