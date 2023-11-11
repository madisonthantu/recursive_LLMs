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
globals.init()
from pathlib import Path
from googleapiclient.errors import HttpError
from googleapiclient import discovery
import requests

def formality_query(payload):
    response = requests.post(globals.API_URL, headers=globals.headers, json=payload)
    return response.json()


def read_lexicon(lex_path):
    lex_df = pd.read_csv(lex_path, sep='\t')
    lex_df.columns = ['word', 'intensity_score']
    lex_df.drop_duplicates(subset=['word'], inplace=True)
    lex_df.set_index('word', inplace=True)
    return lex_df


def read_csv_dataset(text_path):
    text_df = pd.read_csv(text_path, usecols=['summary'])
    text_df['summary'] = text_df['summary'].apply(lambda sentence: re.findall(r'\w+', sentence.lower()))
    return text_df


def toxicity_query(text):
    analyze_request = {
        'comment': { 'text': text },
        'requestedAttributes': {'TOXICITY': {}}
    }
    response = globals.client.comments().analyze(body=analyze_request).execute()
    return response


def preprocess_dataset(df):
    text_df = df.rename(columns=globals.new_col_names)
    text_df['summary'] = text_df['summary'].apply(lambda sentence: re.findall(r'\w+', sentence.lower()))
    return text_df[['id', 'document', 'summary']]



class Measurement:
    def __init__(
            self, 
            data_df,
            dataset_specs,
            rate_limiter_params={'max_calls':64, 'period':10},
            no_samples=1000,
            lex_dir_prefix = 'Data/NRC-Emotion-Intensity-Lexicon/OneFilePerEmotion/',
            DEBUG=False
        ):
        self.data_df = data_df.copy()
        self.rate_limiter = RateLimiter(**rate_limiter_params)
        assert(Path(lex_dir_prefix).exists())
        self.lex_dir_prefix = lex_dir_prefix
        
        assert(k in dataset_specs.keys() for k in ['generation', 'subject']), "Must supply the dataset specs"
        self.config = {
            'subject': dataset_specs['subject'],
            'generation': dataset_specs['generation'],
            'no_samples': no_samples,
            'DEBUG': DEBUG
        }
        
    def compute_coverage(self):
        coverage = self.data_df.apply(lambda x: len(set(x['summary_toks']).intersection(set(x['document_toks']))), axis=1)
        coverage = coverage.divide(self.data_df['document_toks'].apply(lambda x: len(set(x))))
        return coverage.mean()
    
    def compute_compression_ratio(self):
        ratio = self.data_df['summary'].apply(lambda x: len(x.split(" "))).divide(self.data_df['document'].apply(lambda x: len(x.split(" "))))
        return ratio.mean()
    
    def compute_summary_token_distribution(self):
        fdist = FreqDist(tok for tok in list(chain.from_iterable(self.data_df['summary_toks'])))
        return fdist
    
    
    def evaluate_formality(self):
        print("Evaluating formality ...")
        sample_idxs = globals.rng.choice(self.data_df.shape[0], size=self.config['no_samples'], replace=False)
        formality_scores = np.zeros(self.config['no_samples'])
        print("Evaluating formality ...")
        for idx in tqdm(range(self.config['no_samples'])):
            with self.rate_limiter:
                response = formality_query({
                    "inputs": self.data_df.iloc[sample_idxs[idx]]['summary']
                })
                try:
                    formality_scores[idx] = response[0][0]['score']
                except:
                    assert('error' in response.keys())
                    print(f"Formality Eval - Time limit exceeded, sleeping for 10sec, No. samples evaluated = {idx}")
                    time.sleep(10)
                    idx -= 1
        if self.config['DEBUG']:
            return formality_scores, sample_idxs
        return formality_scores
    
    
    def evaluate_toxicity(self):
        print("Evaluating toxicity ...")
        sample_idxs = globals.rng.choice(self.data_df.shape[0], size=self.config['no_samples'], replace=False).astype(int)
        toxicity_scores = np.zeros(self.config['no_samples'])
        print("Evaluating toxicity")
        for idx in tqdm(range(self.config['no_samples'])):
            with self.rate_limiter:
                try:
                    response = toxicity_query(self.data_df.iloc[int(sample_idxs[idx])]['summary'])
                    toxicity_scores[idx] = response['attributeScores']['TOXICITY']['summaryScore']['value']
                except HttpError:
                    print(f"Toxicity Eval - Time limit exceeded, sleeping for 69sec, No. samples evaluated = {idx}")
                    time.sleep(69)
                    idx -= 1
        if self.config['DEBUG']:
            return toxicity_scores, sample_idxs
        return toxicity_scores
    
    
    def evaluate_emotion_intensity(
            self,
            lex_dir_suffix = '-NRC-Emotion-Intensity-Lexicon-v1.txt', 
            lex_names = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']
        ):
        """
        What it do: 
            Create a dataframe where each column corresponds to one of the 8 emotions, 
            the value corresponds to the sum of intensity scores divided by the 
            number of tokens in the summary.
        """
        print("Evaluating emotion intensity ...")
        df = preprocess_dataset(self.data_df)
        emot_scores_df = pd.DataFrame()
        summ_tok_count = df['summary'].apply(lambda x: len(x)).to_numpy()
        weighted_avg = np.zeros(df.shape[0])
        
        print("Evaluating emotion intensity ...")
        for LEX in tqdm(lex_names):
            lex_path = self.lex_dir_prefix + LEX + lex_dir_suffix
            lex_df = read_lexicon(lex_path)
            score_var, cnt_var = f"{LEX}_score_avg", f"{LEX}_tok_cnt"
            res = df['summary'].apply(lambda toks: lex_df.index.str.fullmatch('|'.join(toks)))
            emot_scores_df[score_var] = res.apply(lambda emot_toks: lex_df[emot_toks]['intensity_score'].mean()).fillna(0)
            emot_scores_df[cnt_var] = np.stack(res.values, dtype=int).sum(axis=1)
            w = np.divide(emot_scores_df[cnt_var].to_numpy(), summ_tok_count) * emot_scores_df[score_var].to_numpy()
            weighted_avg = np.add(weighted_avg, w)
            
        emot_scores_df = emot_scores_df.fillna(0)
        emot_scores_df['num_summary_tokens'] = summ_tok_count
        emot_scores_df["weighted_avg"] = weighted_avg
        emot_scores_df['id'] = df['id']
        emot_scores_df.set_index('id')
            
        return emot_scores_df.iloc[:,::-1]
        
        
    def measure(self):
        measurements = {}
        
        measurements['samples'] = self.data_df.iloc[globals.sample_idxs]['summary']
        measurements['sample_idxs'] = globals.sample_idxs
        
        self.data_df['document_toks'] = self.data_df['document'].apply(lambda sentence: re.findall(r'\w+', sentence.lower()))
        self.data_df['summary_toks'] = self.data_df['summary'].apply(lambda sentence: re.findall(r'\w+', sentence.lower()))
        
        measurements['coverage'] = self.compute_coverage()
        measurements['compression_ratio'] = self.compute_compression_ratio()
        measurements['summary_token_distribution'] = self.compute_summary_token_distribution()
        
        formality_eval = self.evaluate_formality()
        toxicity_eval = self.evaluate_toxicity()
        
        if self.config['DEBUG']:
            measurements['formality_scores'], measurements['formality_sample_idxs'] = formality_eval
            measurements['toxicity_scores'], measurements['toxicity_sample_idxs'] = toxicity_eval
        else:
            measurements['formality_scores'] = formality_eval
            measurements['toxicity_scores'] = toxicity_eval
        measurements['formality_mean'] = measurements['formality_scores'].mean()
        measurements['toxicity_mean'] = measurements['toxicity_scores'].mean()
        
        emot_df = self.evaluate_emotion_intensity()
        measurements['emotion_intensity_mean'] = emot_df['weighted_avg'].to_numpy().mean()
        measurements['emotion_intensity_measurements'] = emot_df.to_dict()
        
        return {
            'config': self.config, 
            'metrics': measurements
        }
        