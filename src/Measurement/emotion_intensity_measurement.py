import pandas as pd
import re
import numpy as np


def read_lexicon(lex_path):
    lex_df = pd.read_csv(lex_path, sep='\t')
    lex_df.columns = ['word', 'intensity_score']
    lex_df.drop_duplicates(subset=['word'], inplace=True)
    lex_df.set_index('word', inplace=True)
    return lex_df


def read_text(text_path, col_name):
    text_df = pd.read_csv(text_path, usecols=[col_name])
    text_df[col_name] = text_df[col_name].apply(lambda sentence: re.findall(r'\w+', sentence.lower()))
    return text_df


def process_df(
        input_text_path, 
        col_name = 'summary',
        lex_dir_prefix = '/Users/madisonthantu/Desktop/COMS 6998/Final Project/recursive_LLMs/Data/NRC-Emotion-Intensity-Lexicon/OneFilePerEmotion/',
        lex_dir_suffix = '-NRC-Emotion-Intensity-Lexicon-v1.txt', 
        lex_names = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']
    ):
    """
    What it do: 
        Create a dataframe where each column corresponds to one of the 8 emotions, 
        the value corresponds to the sum of intensity scores divided by the 
        number of tokens in the summary.
    """
    text_df = read_text(input_text_path, col_name)
    
    emot_scores_df = pd.DataFrame()
    emot_scores_df["num_tokens"] = text_df['highlights'].apply(lambda x: len(x))
    
    for LEX in lex_names:
        lex_path = lex_dir_prefix + LEX + lex_dir_suffix
        lex_df = read_lexicon(lex_path)
        
        res = text_df[col_name].apply(lambda toks: lex_df.index.str.fullmatch('|'.join(toks)))
        emot_scores_df[f"{LEX}_score_avg"] = res.apply(lambda emot_toks: lex_df[emot_toks]['intensity_score'].mean())
        emot_scores_df[f"{LEX}_tok_cnt"] = np.stack(res.values, dtype=int).sum(axis=1)
        
    emot_scores_df = emot_scores_df.fillna(0)
    
    weighted_avg = [emot_scores_df.iloc[:, i]*(emot_scores_df.iloc[:, i+1]/emot_scores_df.iloc[:,0])  for i in range(1,emot_scores_df.shape[1], 2)]
    emot_scores_df["weighted_avg"] = pd.concat(weighted_avg, axis=1).sum(axis=1)
        
    return emot_scores_df