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
from random import sample

sys.path.insert(1, '/Users/madisonthantu/Desktop/COMS_6998/Final_Project/recursive_LLMs/Data')
sys.path.append(os.path.abspath("/home/madisonthantu/recursive_LLMs/Data"))
print(platform.platform())
# sys.path.append(os.path.abspath("/Users/madisonthantu/Desktop/COMS_6998/Final_Project/recursive_LLMs/Data"))
# sys.path.append(os.path.abspath("/Users/madisonthantu/Desktop/COMS_6998/Final_Project/recursive_LLMs/Data/NRC-Emotion-Intensity-Lexicon/OneFilePerEmotion"))
# /Users/madisonthantu/Desktop/COMS_6998/Final_Project/recursive_LLMs/Data/NRC-Emotion-Intensity-Lexicon/OneFilePerEmotion/anger-NRC-Emotion-Intensity-Lexicon-v1.txt

from Measurement import globals as globals
globals.init()
from utils import *
init()

from googleapiclient.errors import HttpError
from googleapiclient import discovery
import requests


def formality_query(payload):
    """
    REF: https://huggingface.co/s-nlp/roberta-base-formality-ranker?inference_api=true
    """
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
    """
    REF: https://developers.perspectiveapi.com/s/docs-sample-requests?language=en_US
    """
    analyze_request = {
        'comment': { 'text': text },
        'requestedAttributes': {'TOXICITY': {}}
    }
    response = globals.client.comments().analyze(body=analyze_request).execute()
    return response


def preprocess_dataset(df):
    text_df = df.rename(columns=globals.new_col_names)
    text_df['summary'] = text_df['summary'].apply(lambda sentence: re.findall(r'\w+', sentence.lower()))
    text_df['summary'] = text_df['summary'].apply(lambda sentence: [word for word in sentence if len(word) > 1])
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
        self.lex_dir_prefix = lex_dir_prefix
        
        assert(k in dataset_specs.keys() for k in ['generation', 'subject']), "Must supply the dataset specs"
        if DEBUG:
            no_samples = 10
            globals.sample_idxs = [i for i in range(0,10)]
            print("*** DEBUG MODE **********************")
        self.config = {
            'base_model':dataset_specs['model'],
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
        # Had to change computing of sample_idxs to handle the deletion of rows with `None` values, which occurs with GPT2 output
        # sample_idxs = globals.rng.choice(self.data_df.shape[0], size=self.config['no_samples'], replace=False)
        sample_idxs = sample(self.data_df['id'].tolist(), self.config['no_samples'])
        formality_scores = np.empty(self.config['no_samples'])
        formality_scores[:] = np.nan
        print("Evaluating formality ...")
        # REF: for rate limiting - https://akshayranganath.github.io/Rate-Limiting-With-Python/
        for idx in tqdm(range(self.config['no_samples'])):
            with self.rate_limiter:
                try:
                    input_text = self.data_df[self.data_df['id'] == sample_idxs[idx]]['summary'].values[0]
                except:
                    print(self.data_df[self.data_df['id'] == sample_idxs[idx]])
                    sys.exit()
                try:
                    response = formality_query({
                        "inputs": input_text
                    })
                except requests.exceptions.JSONDecodeError:
                    print("\n", idx, "\n", input_text, "\n", response)
                    sys.exit()
                try:
                    formality_scores[idx] = response[0][0]['score'] if response[0][0]['label']=='formal' else response[0][1]['score']
                    # print(formality_scores[idx], " - ", input_text)
                except:
                    assert('error' in response.keys())
                    print(f"\nFormality Eval - Time limit exceeded, sleeping for 10sec, No. samples evaluated = {idx}")
                    time.sleep(10)
                    idx -= 1
        return formality_scores, sample_idxs
    
    
    def evaluate_toxicity(self):
        print("Evaluating toxicity ...")
        # Had to change computing of sample_idxs to handle the deletion of rows with `None` values, which occurs with GPT2 output
        # sample_idxs = globals.rng.choice(self.data_df.shape[0], size=self.config['no_samples'], replace=False).astype(int)
        sample_idxs = sample(self.data_df['id'].tolist(), self.config['no_samples'])
        toxicity_scores = np.empty(self.config['no_samples'])
        toxicity_scores[:] = np.nan
        print("Evaluating toxicity")
        for idx in tqdm(range(self.config['no_samples'])):
            # REF: for rate limiting - https://akshayranganath.github.io/Rate-Limiting-With-Python/
            with self.rate_limiter:
                try:
                    print(sample_idxs[idx])
                    input_text = self.data_df[self.data_df['id'] == sample_idxs[idx]]['summary'].values[0]
                    response = toxicity_query(input_text)
                    toxicity_scores[idx] = response['attributeScores']['TOXICITY']['summaryScore']['value']
                    # print(toxicity_scores[idx], " - ", input_text)
                except HttpError:
                    print(f"\nToxicity Eval - Time limit exceeded, sleeping for 10sec, No. samples evaluated = {idx}")
                    time.sleep(10)
                    idx -= 1
        return toxicity_scores, sample_idxs
    
    
    def evaluate_emotion_intensity(
            self,
            lex_dir_suffix = '-NRC-Emotion-Intensity-Lexicon-v1.txt', 
            lex_names = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust'],
            root_dir = '/Users/madisonthantu/Desktop/COMS_6998/Final_Project/recursive_LLMs/'
        ):
        """
        What it do: 
            Create a dataframe where each column corresponds to one of the 8 emotions, 
            the value corresponds to the sum of intensity scores divided by the 
            number of tokens in the summary.
        """
        df = preprocess_dataset(self.data_df)
        emot_scores_df = pd.DataFrame()
        summ_tok_count = df['summary'].apply(lambda x: len(x)).to_numpy()
        
        print("Evaluating emotion intensity ...")
        
        emot_score_vars = [f"{LEX}_score_sum" for LEX in lex_names]
        emot_tok_cnt_vars = [f"{LEX}_tok_cnt" for LEX in lex_names]
        
        for i, LEX in tqdm(enumerate(lex_names)):
            print(f"\t... Evaluating {LEX}")
            lex_path = root_dir + self.lex_dir_prefix + LEX + lex_dir_suffix
            lex_df = read_lexicon(lex_path)
            assert(LEX in emot_score_vars[i] and LEX in emot_tok_cnt_vars[i])
            score_var, cnt_var = emot_score_vars[i], emot_tok_cnt_vars[i]
            res = df['summary'].apply(lambda toks: lex_df.index.str.fullmatch('|'.join(toks)))
            emot_scores_df[score_var] = res.apply(lambda emot_toks: lex_df[emot_toks]['intensity_score'].sum()).fillna(0)
            emot_scores_df[cnt_var] = np.stack(res.values, dtype=int).sum(axis=1)
            
        emot_scores_df['num_summary_tokens'] = summ_tok_count
        emot_intensity_sum = emot_scores_df[emot_score_vars].sum(axis=1)
        emot_tok_cnt_sum = emot_scores_df[emot_tok_cnt_vars].sum(axis=1)
        intensity_avg = emot_intensity_sum.divide(emot_tok_cnt_sum)
        emot_scores_df["intensity_ratio"] = intensity_avg * emot_tok_cnt_sum.divide(emot_scores_df['num_summary_tokens'])
        emot_scores_df = emot_scores_df.fillna(0)
        
        emot_scores_df['id'] = df['id']
        emot_scores_df.set_index('id')
            
        return emot_scores_df.iloc[:,::-1]
        
        
    def measure(self):
        measurements = {}
        
        measurements['samples'] = list(self.data_df.iloc[globals.sample_idxs]['summary'])
        measurements['sample_idxs'] = list(globals.sample_idxs)
        
        # print(type(self.data_df['document'].values))
        self.data_df['document_toks'] = self.data_df['document'].apply(lambda sentence: re.findall(r'\w+', sentence.lower()))
        self.data_df['summary_toks'] = self.data_df['summary'].apply(lambda sentence: re.findall(r'\w+', sentence.lower()))
        
        measurements['average_doc_toks'] = sum(self.data_df['document_toks'].apply(lambda doc: len(doc))) / self.data_df.shape[0]
        measurements['average_summ_toks'] = sum(self.data_df['summary_toks'].apply(lambda doc: len(doc))) / self.data_df.shape[0]
        print("measurements['average_doc_toks']\n\t", measurements['average_doc_toks'])
        print("measurements['average_summ_toks']\n\t", measurements['average_summ_toks'])
        
        measurements['coverage'] = self.compute_coverage().item()
        measurements['compression_ratio'] = self.compute_compression_ratio().item()
        measurements['summary_token_distribution'] = dict(self.compute_summary_token_distribution())
        
        formality_eval = self.evaluate_formality()
        toxicity_eval = self.evaluate_toxicity()
        
        formality_scores, formality_sample_idxs = formality_eval
        measurements['formality_mean'] = float(np.nanmean(formality_scores))
        measurements['formality_scores'] = list(formality_scores)
        measurements['formality_sample_idxs'] = list(formality_sample_idxs)
        
        toxicity_scores, toxicity_sample_idxs = toxicity_eval
        measurements['toxicity_mean'] = float(np.nanmean(toxicity_scores))
        measurements['toxicity_scores'] = list(toxicity_scores)
        measurements['toxicity_sample_idxs'] = list(toxicity_sample_idxs)
        
        emot_df = self.evaluate_emotion_intensity()
        measurements['intensity_ratio_avg'] = emot_df['intensity_ratio'].to_numpy().mean().item()
        measurements['emotion_intensity_measurements'] = emot_df.to_dict()
        
        return {
            'config': self.config, 
            'metrics': measurements
        }
        