import src.Measurement.globals as globals
from src.Measurement.measurement import *
globals.init()
print(globals.API_URL)

TRANSFORMERS_NO_ADVISORY_WARNINGS=1

from src.utils import *

import os
import sys
import argparse
from tqdm import tqdm
import pickle
import numpy as np
import logging

from datasets import concatenate_datasets, load_dataset, DatasetDict, Dataset
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
import torch
import platform
import evaluate
from transformers import AutoTokenizer, BartForConditionalGeneration, T5ForConditionalGeneration, pipeline
print(platform.platform())

from huggingface_hub import login


def preprocess_encoder_decoder(examples, padding='max_length', prefix = "summarize: "):
    inputs = [prefix + doc for doc in examples["document"]]
    model_inputs = tokenizer(inputs, max_length=globals.max_src_length[args['dataset']], padding='max_length', truncation=True)#, return_tensors='pt')
    labels = tokenizer(text_target=examples["summary"], max_length=globals.max_target_length[args['dataset']], padding=padding, truncation=True)#, return_tensors='pt')
    # if padding == "max_length":
    #     labels["input_ids"] = [
    #         [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
    #     ]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    rouge_res = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    rouge_res["summary_length"] = np.mean(prediction_lens)
    results = {k: round(v, 4) for k, v in rouge_res.items()}
    
    bert_res = bert_score.compute(predictions=decoded_preds, references=decoded_labels, lang="en")
    results.update(bert_res)
    
    return results


def generate_new_synthetic_dataset(model, args):
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
    all_data = concatenate_datasets([tokenized_data[k] for k in tokenized_data.keys()])
    inputs = ["summarize: " + doc for doc in all_data["document"]]
    # summaries = summarizer(inputs, min_length=15, max_length=globals.max_target_length[args['dataset']])
    dataset_inputs = ListDataset(inputs)
    summaries = []
    for out in tqdm(summarizer(dataset_inputs, min_length=15, max_length=globals.max_target_length[args['dataset']])):
        summaries.append(out)
    synthetic_df = pd.DataFrame({
        'id': all_data['id'],
        'document': all_data['document'],
        'summary': [d['summary_text'] for d in summaries]
        })
    return synthetic_df

if __name__ == "__main__":
    """
    What it do:
        1. fine tune the model
        2. compute_metrics results and save
        2. save the new checkpoint
        3. generate new synthetic dataset
        4. perform measurement analysis on the new dataset
    """
    init()
    parser = argparse.ArgumentParser(description='running parameter experiments')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--base_model', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--num_generations', type=int, default=3)
    parser.add_argument('--data_dir', type=str, default='/Users/madisonthantu/Desktop/COMS 6998/Final Project/recursive_LLMs/Data')
    parser.add_argument('--DEBUG', type=bool, default=False)
    

    parsed_args = parser.parse_args()
    args = vars(parsed_args)
    
    base_models = {
        't5': 't5-small',
    }
    ModelConstructor = {
        't5':T5ForConditionalGeneration,
    }
    learning_rates = {
        't5': 1e-4,
    }
    dataset_configs = {
        'news': ["cnn_dailymail", "2.0.0"],
        'reddit': ["reddit_tifu", 'long'],
        'dialogue': ['samsum']
    }

    assert(args['base_model'] in base_models.keys()), "Invalid 'base_model' supplied"
    assert(args['dataset'] in dataset_configs.keys()), "Invalid 'dataset' suplied"

    base_path = make_output_base_path_str(args)
    # print("base_path =", base_path)
    synthetic_data_path = os.path.join(base_path, 'synthetic_data')
    if os.path.isdir(synthetic_data_path):
        print(f"The supplied directory for the synthetic data, {[synthetic_data_path]}, already exists. Do you wish to overwrite this directory's contents? [y/n]: ")
        if str(input()).lower() != "y":
            sys.exit()
    else:
        os.makedirs(synthetic_data_path)
            

    dataset_key, dataset_config = args['dataset'], dataset_configs[args['dataset']]
    base_model = args['base_model']
    base_model_checkpoint = base_models[base_model]

    tokenizer = AutoTokenizer.from_pretrained(base_model_checkpoint)
    tokenizer.pad_token = tokenizer.unk_token
    # print(tokenizer)

    for gen_num in tqdm(range(args['num_generations'])):
        print(f"Generation {gen_num}")
        generation_path = os.path.join(base_path, f"generation{gen_num}")
        # assert(not os.path.exists(generation_path))
        if not os.path.exists(generation_path):
            os.makedirs(generation_path)
            os.makedirs(os.path.join(generation_path, 'results'))
        
        if gen_num == 0:
            model_checkpoint = base_model_checkpoint
            dataset = load_huggingface_dataset(dataset_config, args)
            
        else:
            model_checkpoint = os.path.join(base_path, f"generation{gen_num-1}")
            dataset = load_synthetic_dataset(synthetic_data_path, "synthetic_data.pkl")
        
        tokenized_data = dataset.map(preprocess_encoder_decoder, batched=True)
        tokenized_data = train_val_test_split(tokenized_data)
        
        assert(tokenized_data['train'] != tokenized_data['test'])
        assert(tokenized_data['train'] != tokenized_data['validation'])
        assert(tokenized_data['validation'] != tokenized_data['test'])
        
        print("Executing DataCollatorForSeq2Seq ...")
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer, 
            model=model_checkpoint,
            label_pad_token_id=tokenizer.pad_token,
        )
        
        model = ModelConstructor[base_model].from_pretrained(model_checkpoint)
        # model.resize_token_embeddings(len(tokenizer))
        # model.generation_config.max_new_tokens = globals.max_target_length[args['dataset']]
        
        rouge = evaluate.load("rouge")
        bert_score = evaluate.load("bertscore")
        
        print("Executing Seq2SeqTrainingArguments ...")
        training_args = Seq2SeqTrainingArguments(
            output_dir=generation_path,
            evaluation_strategy="epoch",
            learning_rate=learning_rates[base_model],
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=1,
            predict_with_generate=True,
            generation_max_length = globals.max_target_length[args['dataset']],
            push_to_hub=True,
        )
        print("Exeuting Seq2SeqTrainer ...")
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_data["train"],
            eval_dataset=tokenized_data["validation"],
            # tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        # trainer.train()
        trainer.save_model()
        
        print("Executing trainer.evaluate(<test_data>) ...")
        # test_data_eval_results = trainer.evaluate(eval_dataset=tokenized_data["test"])
        # print(test_data_eval_results)
        
        # Generate new synthetic dataset
        print("Generating new synthetic dataset ...")
        new_dataset = generate_new_synthetic_dataset(generation_path, args)
        dataset_specs = {
            'generation':gen_num, 
            'subject':args['dataset']
        }
        print("Collecting measurements on new synthetic dataset ...")
        synthetic_dataset_measurements = Measurement(new_dataset, dataset_specs, DEBUG=True)
        synthetic_dataset_results = synthetic_dataset_measurements.measure()
        
        files_to_save = {
            # os.path.join(generation_path, 'results', 'test_data_eval_results.pkl'): test_data_eval_results,
            os.path.join(generation_path, 'results', 'config.pkl'): synthetic_dataset_results['config'],
            os.path.join(generation_path, 'results', 'measurements.pkl'): synthetic_dataset_results['metrics'],
            os.path.join(synthetic_data_path, 'synthetic_data.pkl'): new_dataset # Over-writing dataset generated during previous iteration
        }
        save_files_to_pkl(files_to_save)
        if gen_num == 0:
            pickle.dump(new_dataset, open(os.path.join(generation_path, 'synthetic_data.pkl'), "wb"))
        
        print(f"Generation {gen_num+1}/{args['num_generations']} done! ... {globals.emoji_dict[gen_num]} ")
        