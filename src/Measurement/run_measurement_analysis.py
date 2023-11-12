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

sys.path.insert(1, '../Data')
sys.path.append(os.path.abspath("/home/madisonthantu/recursive_LLMs/Data"))
print(platform.platform())

import src.Measurement.globals as globals
from Measurement import Measurement
globals.init()

from googleapiclient.errors import HttpError
from googleapiclient import discovery
import requests

def __main__():
    