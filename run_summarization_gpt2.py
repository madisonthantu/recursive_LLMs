#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import logging
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Optional
import json
import torch
from torch.nn.functional import softmax

from tqdm import tqdm

import datasets
import evaluate
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets import load_dataset
from filelock import FileLock

import transformers
from transformers import (
    DataCollatorForLanguageModeling, 
    GPT2Tokenizer, 
    GPT2LMHeadModel, 
    Trainer, 
    TrainingArguments, 
    AutoConfig,
    HfArgumentParser,
    set_seed,
)

from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, is_offline_mode, send_example_telemetry
from transformers.utils import logging as HF_log
from transformers.utils.versions import require_version

# REF: https://discuss.huggingface.co/t/how-to-turn-wandb-off-in-trainer/6237
os.environ["WANDB_DISABLED"] = "true"



# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.36.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/summarization/requirements.txt")

logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
                "the model's position embeddings."
            )
        },
    )
    base_model: str = field(
        default=None,
        metadata={
            "help": (
                "Base model string."
            )
        },
    )
    gen_num: int = field(
        default=None,
        metadata={
            "help": (
                "Generation number."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    lang: Optional[str] = field(default=None, metadata={"help": "Language id for summarization."})

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    text_column: Optional[str] = field(
        default='document',
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    summary_column: Optional[str] = field(
        default='summary',
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "An optional input evaluation data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
            )
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`. "
                "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                "during ``evaluate`` and ``predict``."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )

    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The token to force as the first generated token after the decoder_start_token_id. "
                "Useful for multilingual models like mBART where the first generated token"
                "needs to be the target language token (Usually it is the target language token)"
            )
        },
    )
    DEBUG: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "DEBUG option. Set to True to avoid certail assertion check (w.r.t. file paths and naming)."
            )
        },
    )
    data_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
        },
    )
    num_epochs: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of training epochs."
            )
        },
    )
    

    def __post_init__(self):
        if (
            self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
            and self.test_file is None
        ):
            raise ValueError("Need either a dataset name or a training, validation, or test file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
            if self.test_file is not None:
                extension = self.test_file.split(".")[-1]
                assert extension in ["csv", "json"], "`test_file` should be a csv or a json file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


dataset_names = [
    "dialogue",
    "news",
    "reddit"
]

model_names = [
    'gpt2'
]

summarization_name_mapping = {
    "cnn_dailymail": ("article", "highlights"),
    "samsum": ("dialogue", "summary"),
    "reddit": ("documents", "tldr")
}

bos = '<|endoftext|>'
eos = '<|EOS|>'
pad = '<|pad|>'

special_tokens_dict = {
    'eos_token': eos, 
    'bos_token': bos, 
    'pad_token': pad
}
max_target_lenghts = {
    'news':175,
    'reddit':100,
    'dialogue':100
}

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead.",
            FutureWarning,
        )
        if model_args.token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        model_args.token = model_args.use_auth_token

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_summarization", model_args, data_args)
    
    assert(data_args.dataset_name in dataset_names), "Must supply valid <dataset_name>"
    
    assert(not os.path.isfile(os.path.join(training_args.output_dir, 'train_results.json')) if training_args.do_train else 1)
    assert(not os.path.isfile(os.path.join(training_args.output_dir, 'eval_results.json')) if training_args.do_eval else 1)
    assert(not os.path.isfile(os.path.join(training_args.output_dir, 'predict_results.json')) if training_args.do_predict else 1)
    
    if data_args.DEBUG == False:
        assert(data_args.dataset_name in data_args.test_file if data_args.test_file is not None else 1), "<test_data> path and <dataset_name> do not match"
        assert(data_args.dataset_name in data_args.validation_file if data_args.validation_file is not None else 1), "<validation_data> path and <dataset_name> do not match"
        assert(data_args.dataset_name in data_args.train_file if data_args.train_file is not None else 1), "<training_data> path and <dataset_name> do not match"
        # assert(model_names[0] in training_args.output_dir or model_names[1] in training_args.output_dir or model_names[2] in training_args.output_dir), "Must include <base_model_name> in output directory path"
        assert(model_args.base_model in training_args.output_dir), "Must include <base_model_name> in output directory path"
        
        assert(f"gen{model_args.gen_num}" in training_args.output_dir), "Must include <generation_number> in output directory path"
        # if data_args.do_predict and not data_args.do_train 
        if os.path.isdir(training_args.output_dir):
            assert(f"gen{model_args.gen_num}" in model_args.model_name_or_path)
        
    if model_args.gen_num == 0:
        assert("initial_datasets" in data_args.train_file if data_args.train_file is not None else 1)
        assert("initial_datasets" in data_args.validation_file if data_args.validation_file is not None else 1)
        assert("initial_datasets" in data_args.test_file if data_args.test_file is not None else 1)
        assert("initial_datasets" in data_args.data_path if data_args.data_path is not None else 1)
    else:
        assert("synthetic_datasets" in data_args.train_file if data_args.train_file is not None else 1)
        assert("synthetic_datasets" in data_args.validation_file if data_args.validation_file is not None else 1)
        assert("synthetic_datasets" in data_args.test_file if data_args.test_file is not None else 1)
        assert("synthetic_datasets" in data_args.data_path if data_args.data_path is not None else 1)
        
    assert(data_args.dataset_name in training_args.output_dir.lower()), "Must include <dataset_name> in output directory path"
    assert(f"gen{model_args.gen_num}" in training_args.output_dir), "Generation no. must be in <output_dir>"
    assert(data_args.text_column == 'document'), "Must specify the name of the <text_column>"
    assert(data_args.summary_column == 'summary'), "Must specify the name of the <summary_column>"
    assert(model_args.base_model in model_names), "Must supply name of base model to <base_model>"

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    if training_args.do_train and last_checkpoint is None and model_args.gen_num != 0:
        assert(str(model_args.gen_num-1) in model_args.model_name_or_path), "Previous generation number should be in path for model checkpoint."

    set_seed(training_args.seed)
    
    data_files = {}
    if data_args.data_path is not None:
        data_files["train"] = os.path.join(data_args.data_path, 'training_data.csv')
        data_files["validation"] = os.path.join(data_args.data_path, 'validation_data.csv')
        data_files["test"] = os.path.join(data_args.data_path, 'full_data.csv')
        extension = "csv"
    else:
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
    
    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        cache_dir=model_args.cache_dir,
        token=model_args.token,
    )
    
    tokenizer = GPT2Tokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    tokenizer.padding_side = 'left'
    target_length = max_target_lenghts[data_args.dataset_name]
    separator = " TL;DR "
    separator_ids = tokenizer.encode(separator)
    separator_length = len(separator_ids)
    separator_attention_mask = [1] * separator_length
    separator_length = len(tokenizer.tokenize(separator))
    global_max_length = tokenizer.model_max_length
    src_length = global_max_length - separator_length - target_length
    
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        trust_remote_code=model_args.trust_remote_code,
    )
    model = GPT2LMHeadModel.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    model.resize_token_embeddings(len(tokenizer))
    
    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        column_names = raw_datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # Get the column names for input/target.
    dataset_columns = summarization_name_mapping.get(data_args.dataset_name, None)
    if data_args.text_column is None:
        text_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        text_column = data_args.text_column
        if text_column not in column_names:
            raise ValueError(
                f"--text_column' value '{data_args.text_column}' needs to be one of: {', '.join(column_names)}"
            )
    if data_args.summary_column is None:
        summary_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        summary_column = data_args.summary_column
        if summary_column not in column_names:
            raise ValueError(
                f"--summary_column' value '{data_args.summary_column}' needs to be one of: {', '.join(column_names)}"
            )

    def preprocess_function(examples): 
        model_input = torch.full((1, tokenizer.model_max_length), tokenizer.pad_token_id)
        doc_ids = tokenizer.encode(bos + ' ' + examples[text_column], padding='do_not_pad', max_length=src_length, truncation=True, return_tensors='pt')
        summ_ids = tokenizer.encode(" TL;DR " + examples[summary_column], padding='do_not_pad', max_length=target_length-1, truncation=True, return_tensors='pt')
        in_put = torch.cat((doc_ids, summ_ids, torch.Tensor([[tokenizer.eos_token_id]])), axis=1)
        model_input[0,model_input.shape[1]-in_put.shape[1]:] = in_put
            
        input_ids = model_input
        attention_mask = torch.where(model_input!=tokenizer.pad_token_id, 1, 0)
        return {
            'input_ids':input_ids,
            'attention_mask':attention_mask
        }
    
    def preprocess_generation_function(example):
        inputs = bos + ' ' + example[text_column]
        targets = example[summary_column]

        model_inputs = tokenizer(inputs, padding='max_length', max_length=src_length, truncation=True)
        model_inputs['input_ids'] = model_inputs['input_ids'] + separator_ids
        model_inputs['attention_mask'] = model_inputs['attention_mask'] + separator_attention_mask
        # target_ids = tokenizer(targets, padding='max_length', max_length=target_length, truncation=True)
        # model_inputs['label_ids'] = target_ids['input_ids']
        return model_inputs
    
    
    if training_args.do_train:
        train_dataset = raw_datasets["train"]
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=False,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )

    if training_args.do_eval:
        eval_dataset = raw_datasets["validation"]
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=False,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

    if training_args.do_predict:
        predict_dataset = raw_datasets["test"]
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_generation_function,
                batched=False,
                # num_proc=data_args.preprocessing_num_workers,
                # remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )
            predict_dataset.set_format("pt", columns=["input_ids", "attention_mask"], output_all_columns=True) 
            print(predict_dataset)
            
    data_collator = DataCollatorForLanguageModeling(
        tokenizer,
        mlm=False
    )

    # Metric
    metric = evaluate.load("rouge")

    def postprocess_text(preds, labels):
        labels = [label[label.find("TL;DR")+len("TL;DR"):] for label in labels]
        preds = [pred[pred.find("TL;DR")+len("TL;DR"):] for pred in preds]
        # print("\nlabels[0]\n\t", labels[0])
        # print("\npreds[0]\n\t", preds[0])
        labels = [label.strip() for label in labels]
        preds = [pred.strip() for pred in preds]
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
        return preds, labels

    # REF: https://discuss.huggingface.co/t/cuda-out-of-memory-when-using-trainer-with-compute-metrics/2941/10
    def preprocess_logits_for_metrics(logits, labels):
        """
        Original Trainer may have a memory leak. 
        This is a workaround to avoid storing too many tensors that are not needed.
        """
        pred_ids = torch.argmax(logits, dim=-1)
        return pred_ids, labels
    
    def compute_metrics(eval_preds):
        pred_ids = eval_preds.predictions
        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]
        pred_ids = np.squeeze(pred_ids)
        label_ids = eval_preds.label_ids
        # Replace -100s used for padding as we can't decode them
        pred_ids = np.where(pred_ids != -100, pred_ids, tokenizer.pad_token_id)
        pred_strs = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_ids = np.squeeze(label_ids)
        label_ids = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)
        label_strs = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        decoded_preds, decoded_labels = postprocess_text(pred_strs, label_strs)
        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [len(tokenizer.tokenize(dec_pred)) for dec_pred in decoded_preds]
        result["gen_len"] = np.mean(prediction_lens)
        return result
    
    def compute_generation_metrics(pred_strs, label_strs):
        decoded_preds, decoded_labels = postprocess_text(pred_strs, label_strs)
        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [len(tokenizer.tokenize(pred)) for pred in pred_strs]
        result["gen_len"] = np.mean(prediction_lens)
        return result
    
    training_args.num_train_epochs = (
        data_args.num_epochs if data_args.num_epochs is not None else training_args.num_train_epochs
    )

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        if isinstance(eval_dataset, dict):
            metrics = {}
            for eval_ds_name, eval_ds in eval_dataset.items():
                dataset_metrics = trainer.evaluate(eval_dataset=eval_ds, metric_key_prefix=f"eval_{eval_ds_name}")
                metrics.update(dataset_metrics)
        else:
            metrics = trainer.evaluate(metric_key_prefix="eval")
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        
    
    def postprocess_generations(pred_strs):
        preds = [pred[pred.find("TL;DR")+len("TL;DR"):] for pred in pred_strs]
        preds = [pred.strip() for pred in preds]
        return preds
        
    
    if training_args.do_predict:
        logger.info("*** Predict ***")
        logger.setLevel(logging.WARNING)
        # print(f"logger.level = {logger.level}")
        HF_log.set_verbosity_warning()
        # model = model.to('cpu')
        pred_strs = []
        label_strs = predict_dataset[summary_column]
        
        n = 5
        for i in tqdm(range(0, len(predict_dataset), n), total=int(len(predict_dataset)/n)):
            sample_dataset = predict_dataset[i:i+n]
            # print(i)
            # print(sample_dataset)
            output_ids = model.generate(
                input_ids=sample_dataset['input_ids'].to("cuda"),
                attention_mask=sample_dataset['attention_mask'].to("cuda"),
                do_sample=False,
                early_stopping=True,
                max_new_tokens = target_length
            )
            pred_strs += tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            # print(pred_strs[-1])
            
        pred_strs = postprocess_generations(pred_strs)
        metrics = compute_generation_metrics(pred_strs, label_strs)
        metrics["predict_samples"] = len(predict_dataset)

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            # if training_args.predict_with_generate:
            # predictions = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            synthetic_data_df = predict_dataset.select_columns(['document','id']).to_pandas()
            print(synthetic_data_df.head(5))
            synthetic_data_df['summary'] = pred_strs
            synthetic_data_df['summary'] = synthetic_data_df['summary'].apply(lambda summ: summ.strip())
            print(synthetic_data_df.head(5))

            output_synthetic_data_path = os.path.join("Data/synthetic_datasets", str(data_args.dataset_name), model_args.base_model, f"gen{model_args.gen_num}")
            if not os.path.exists(output_synthetic_data_path):
                os.makedirs(output_synthetic_data_path)
            synthetic_data_df.to_csv(os.path.join(output_synthetic_data_path, 'full_data.csv'), index=False)
            config = {
                'base_model': model_args.base_model,
                'gen_num': model_args.gen_num,
                'dataset_name': data_args.dataset_name,
                'DEBUG': data_args.DEBUG,
                'num_samples': synthetic_data_df.shape[0]
            }
            with open(os.path.join(output_synthetic_data_path, 'config.json'), 'w') as f:
                json.dump(config, f)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "summarization"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if data_args.lang is not None:
        kwargs["language"] = data_args.lang

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
