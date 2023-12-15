import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go


def compute_qualitative_summary_statistics_df(
    metric_list, 
    keys,
    variables = ['average_summ_toks', 'coverage', 'compression_ratio', 'toxicity_mean', 'formality_mean', 'intensity_ratio_avg', 'top_n_emotions', 'top_n_emotion_scores'],
    n=3
    ):
    res_df = pd.DataFrame(columns=['model', 'generation', 'dataset'] + variables)
    for i, metrics in enumerate(metric_list):
        res = {}
        res['model'] = keys[i][0]
        res['dataset'] = keys[i][-1]
        res['generation'] = keys[i][1]
        if 'average_summ_toks' in variables:
            res['average_summ_toks'] = round(metrics['average_summ_toks'], 2)
        if 'coverage' in variables:
            res['coverage'] = round(metrics['coverage'], 2)
        if 'compression_ratio' in variables:
            res['compression_ratio'] = round(metrics['compression_ratio'], 2)
        if 'toxicity_mean' in variables:
            res['toxicity_mean'] = round(metrics['toxicity_mean'],2)
        if 'formality_mean' in variables:
            res['formality_mean'] = round(metrics['formality_mean'],2)
        if 'intensity_ratio_avg' in variables:
            res['intensity_ratio_avg'] = round(metrics['intensity_ratio_avg'],2)
        if 'top_n_emotions' in variables:
            lex_names = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']
            df = pd.DataFrame(metrics['emotion_intensity_measurements'])
            emotion_score_vars = [lex+'_score_sum' for lex in lex_names]
            emotion_score_vars
            top_n_emots = df[emotion_score_vars].mean(axis=0).nlargest(n=3, keep='first')
            res['top_n_emotions'] = ", ".join([s[:s.find('_')] for s in top_n_emots.index])
            res['top_n_emotion_scores'] = top_n_emots.round(decimals=2).values
            
        res_df.loc[len(res_df)] = res
    return res_df

def plot_line(data_df, y, x, color='model', x_label=None, y_label=None, title=""):
    x_axis = list(data_df[x].unique())
    models = list(data_df[color].unique())
    fig = go.Figure()
    if 'baseline' in models:
        models.remove('baseline')
        x_axis.remove('baseline')
        fig.add_trace(go.Scatter(x=x_axis,
                               y=[data_df[data_df['model'] == 'baseline'][y].values[0]] * len(x_axis), 
                               mode='lines',
                               marker=dict(size=7),
                               opacity=.5,
                               name='Baseline'
                    ))
    for model in models:
        fig.add_trace(go.Scatter(x=x_axis,
                               y=data_df[data_df['model'] == model][y].values,
                               connectgaps=True,
                               marker=dict(size=7),
                               name=model
                    ))
    fig.update_layout(title=title, 
                      showlegend=True,
                      xaxis_title=x_label,
                      yaxis_title=y_label,
                      )
    return fig


def compute_token_distribution_df(data_dict):
    df = pd.DataFrame(columns=['model', 'subject', 'generation', 'vocab_size', 'avg_tok_occurrence', 'min_occurring_tok', 'max_occurring_tok'])
    for model in data_dict.keys():
        for gen in data_dict[model].keys():
            tok_dist = data_dict[model][gen]['metrics']['summary_token_distribution']
            toks = list(tok_dist.keys())
            occurs = list(tok_dist.values())
            df.loc[len(df)] = {
                'model':model,
                'subject':data_dict[model][gen]['subject'],
                'generation':gen,
                'vocab_size':len(tok_dist.keys()),
                'avg_tok_occurrence':sum(tok_dist.values())/len(tok_dist.keys()),
                'min_occurring_tok':(toks[np.argmin(occurs)], min(occurs)),
                'max_occurring_tok':(toks[np.argmax(occurs)], max(occurs))
            }
    return df
        
        
def plot_violin(data_df, y, x, color, x_label=None, y_label=None, title=""):
    x_label = x if x_label is None else x_label
    y_label = x if y_label is None else y_label
    fig = px.violin(data_df, y=y, x=x, color=color, box=True, hover_data=data_df.columns).update_layout(
        xaxis_title=x_label, yaxis_title=y_label, title_text=title
    )
    return fig

import math

def find_max_emot_example(emot_df, data_df, n=1):
    top_emot_samples = emot_df.nlargest(n=n, columns='intensity_ratio')
    top_emot_ids = top_emot_samples['id'].values
    res_df = data_df[data_df['id'].isin(top_emot_ids)].copy()
    res_df['intensity_ratio'] = emot_df[emot_df['id'].isin(top_emot_ids)]['intensity_ratio']
    return res_df

def find_min_emot_example(emot_df, data_df, n=1):
    bottom_emot_samples = emot_df[emot_df['intensity_ratio'] != 0].nsmallest(n=n, columns='intensity_ratio')
    bottom_emot_ids = bottom_emot_samples['id'].values
    res_df = data_df[data_df['id'].isin(bottom_emot_ids)].copy()
    res_df['intensity_ratio'] = emot_df[emot_df['id'].isin(bottom_emot_ids)]['intensity_ratio']
    return res_df

def find_max_toxicity(results, data_df, n=1):
    toxicity_scores = np.array(results['toxicity_scores'])
    args_sorted = np.argsort(toxicity_scores)[::-1]
    toxicity_sample_idxs = np.array(results['toxicity_sample_idxs'])
    ids_sorted = toxicity_sample_idxs[args_sorted]
    res_df = pd.DataFrame(columns=['document', 'id', 'summary', 'toxicity_score'])
    for i in range(args_sorted.size):
        if not np.isnan(toxicity_scores[args_sorted[i]]):
            sample = data_df[data_df['id']==ids_sorted[i]]
            res_df.loc[len(res_df.index)] = [
                sample['document'].values[0],
                sample['id'].values[0],
                sample['summary'].values[0],
                toxicity_scores[args_sorted[i]]
            ] 
            if res_df.shape[0] >= n:
                return res_df
        i += 1
    return res_df

def find_min_toxicity(results, data_df, n=1):
    toxicity_scores = np.array(results['toxicity_scores'])
    args_sorted = np.argsort(toxicity_scores)
    toxicity_sample_idxs = np.array(results['toxicity_sample_idxs'])
    ids_sorted = toxicity_sample_idxs[args_sorted]
    res_df = pd.DataFrame(columns=['document', 'id', 'summary', 'toxicity_score'])
    i = 0
    for i in range(toxicity_scores.size):
        if toxicity_scores[args_sorted[i]] != 0:
            sample = data_df[data_df['id']==ids_sorted[i]]
            res_df.loc[len(res_df.index)] = [
                sample['document'].values[0],
                sample['id'].values[0],
                sample['summary'].values[0],
                toxicity_scores[args_sorted[i]]
            ]
            if res_df.shape[0] >= n:
                return res_df
        i += 1
    return res_df


def find_max_formality(results, data_df, n=1):
    formality_scores = np.array(results['formality_scores'])
    args_sorted = np.argsort(formality_scores)[::-1]
    formality_sample_idxs = np.array(results['formality_sample_idxs'])
    ids_sorted = formality_sample_idxs[args_sorted]
    res_df = pd.DataFrame(columns=['document', 'id', 'summary', 'formality_score'])
    for i in range(formality_scores.size):
        if not np.isnan(formality_scores[args_sorted[i]]):
            sample = data_df[data_df['id']==ids_sorted[i]]
            res_df.loc[len(res_df.index)] = [
                sample['document'].values[0],
                sample['id'].values[0],
                sample['summary'].values[0],
                formality_scores[args_sorted[i]]
            ] 
            if res_df.shape[0] >= n:
                return res_df
        i += 1
    return res_df

def find_min_formality(results, data_df, n=1):
    formality_scores = np.array(results['formality_scores'])
    args_sorted = np.argsort(formality_scores)
    formality_sample_idxs = np.array(results['formality_sample_idxs'])
    ids_sorted = formality_sample_idxs[args_sorted]
    res_df = pd.DataFrame(columns=['document', 'id', 'summary', 'formality_score'])
    i = 0
    for i in range(formality_scores.size):
        if formality_scores[args_sorted[i]] != 0:
            sample = data_df[data_df['id']==ids_sorted[i]]
            res_df.loc[len(res_df.index)] = [
                sample['document'].values[0],
                sample['id'].values[0],
                sample['summary'].values[0],
                formality_scores[args_sorted[i]]
            ]
            if res_df.shape[0] >= n:
                return res_df
        i += 1
    return res_df