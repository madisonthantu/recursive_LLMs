import src.Measurement.globals as globals
globals.init()
print(globals.API_URL)

import os
import sys
import argparse
from tqdm import tqdm
import pickle
import numpy as np


from datasets import concatenate_datasets, load_dataset, DatasetDict, Dataset
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
import torch
import platform
import evaluate
from transformers import AutoTokenizer
print(platform.platform())


from huggingface_hub import login

def init():
    login(token=globals.hug_token)
    

def preprocess_text_function(examples, tokenizer, prefix = "summarize: "):
    inputs = [prefix + doc for doc in examples["document"]]
    model_inputs = tokenizer(inputs, max_length=globals.max_src_length[args['dataset']], truncation=False)
    labels = tokenizer(text_target=examples["summary"], max_length=globals.max_target_length[args['dataset']], truncation=False)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def load_huggingface_dataset(data_config):
    data_dict = load_dataset(*data_config)
    dataset = concatenate_datasets([data_dict[k] for k in data_dict.keys()])
    print(dataset.column_names)
    for col in dataset.column_names:
        if col in globals.new_col_names.keys():
            dataset = dataset.rename_column(col, globals.new_col_names[col])
            print(col, globals.new_col_names[col])
    print(dataset.column_names)
    if args['dataset'] == 'reddit':
        return dataset.select_columns(['document', 'summary'])
    else:
       return dataset.select_columns(['id', 'document', 'summary'])
    
    
def train_val_test_split(dataset):
    data_dict = dataset.train_test_split(test_size=0.3)
    dev_data_dict = data_dict['test'].train_test_split(test_size=0.5)
    print()
    print(data_dict)
    print()
    print(dev_data_dict)
    return DatasetDict({
        'train': data_dict['train'],
        'validation': dev_data_dict['train'],
        'test': dev_data_dict['test']
    })
    

def make_output_base_path_str():
    path = os.path.join(args['output_dir'], args['base_model'], args['dataset'])
    print(f"Output directory path: {path}")
    if os.path.isdir(path):
        print(f"The supplied output directory, {[path]}, already exists. Do you wish to overwrite this directory's contents? [y/n]: ")
        if str(input()).lower() != "y":
            sys.exit()
    return path


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    rouge_res = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    rouge_res["summary_length"] = np.mean(prediction_lens)
    results = {k: round(v, 4) for k, v in rouge_res.items()}
    
    bert_res = bert_score.compute(predictions=decoded_preds, references=decoded_labels)
    results.update({k: round(v,4) for k, v in bert_res.items()})
    
    return results
    
    
if __name__ == "__main__":
    init()
    parser = argparse.ArgumentParser(description='running parameter experiments')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--base_model', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--num_generations', type=int, default=3)
    
    parsed_args = parser.parse_args()
    args = vars(parsed_args)
    
    base_models = {
        't5': 't5-small'   
    }
    dataset_configs = {
        'news': ["cnn_dailymail", "2.0.0"],
        'reddit': ["reddit_tifu", 'long'],
        'dialogue': ['samsum']
    }
    
    assert(args['base_model'] in base_models.keys()), "Invalid 'base_model' supplied"
    assert(args['dataset'] in dataset_configs.keys()), "Invalid 'dataset' suplied"
    
    base_path = make_output_base_path_str()
    print("base_path =", base_path)
    
    dataset_key, dataset_config = args['dataset'], dataset_configs[args['dataset']]
    base_model_checkpoint = base_models[args['base_model']]
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_checkpoint)
    
    for gen_num in tqdm(range(args['num_generations'])):
        print(f"Generation {gen_num}")
        if gen_num == 0:
            dataset = load_huggingface_dataset(dataset_config)
            
        else:
            
            data_dict = train_val_test_split(dataset)
            
            tokenized_data = data_dict.map(lambda data_x: preprocess_text_function(data_x, tokenizer), batched=True)
            tokenized_data = tokenized_data.filter(lambda example: len(example['input_ids']) <= globals.max_src_length[args['dataset']])
            tokenized_data = tokenized_data.filter(lambda example: len(example['labels']) <= globals.max_target_length[args['dataset']])
            
            assert(tokenized_data['train'] != tokenized_data['test'])
            assert(tokenized_data['train'] != tokenized_data['validation'])
            assert(tokenized_data['validation'] != tokenized_data['test'])
            
            data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model_checkpoint)
            
            rouge = evaluate.load("rouge")
            bert_score = evaluate.load("bertscore")