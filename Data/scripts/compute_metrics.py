import sys
import argparse
import os
import pandas as pd
from datasets import concatenate_datasets, load_dataset, DatasetDict, Dataset

import numpy as np
import json
import evaluate

from datasets import load_dataset
import nltk

metric = evaluate.load("rouge")


def postprocess_text(preds, labels):
        preds = [str(pred).strip() for pred in preds]
        labels = [str(label).strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels
    

def compute_metrics(pred_summs, true_summs):
    # Some simple post-processing
    print("\nComputing metrics ...")
    decoded_preds, decoded_labels = postprocess_text(pred_summs, true_summs)
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in result.items()}
    return result


def main():
    parser = argparse.ArgumentParser(description='running dataset preprocessing')
    parser.add_argument('--synth_data_path', type=str)
    parser.add_argument('--init_data_path', type=str)
    parser.add_argument('--file_name', type=str, default='full_data.csv')
    parser.add_argument('--dataset_name', type=str, default='')
    parser.add_argument('--base_model', type=str)
    args = parser.parse_args()
    
    assert(args.dataset_name in args.init_data_path)
    assert(args.dataset_name in args.synth_data_path)
    # assert(args.base_model in args.init_data_path)
    assert(args.base_model in args.synth_data_path)
    
    pred_summs = pd.read_csv(os.path.join(args.synth_data_path, args.file_name)).drop(labels=['document'], axis=1)
    true_summs = pd.read_csv(os.path.join(args.init_data_path, args.file_name)).drop(labels=['document'], axis=1)
    
    summs = pred_summs.join(true_summs, on='id', how='inner', lsuffix='_pred', rsuffix='_true').drop(labels=['id_pred', 'id_true'], axis=1)
    
    print("\n", summs.head(5))
    print(args.synth_data_path)
    
    metric_res = compute_metrics(summs['summary_pred'].values.tolist(), summs['summary_true'].values.tolist())
    
    print(metric_res)
    
    with open(os.path.join(args.synth_data_path, 'metric_results.json'), 'w') as f:
           json.dump(metric_res, f)
    
    print("Writing metrics ...\n\n")
    
if __name__ == "__main__":
    main()