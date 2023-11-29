---
base_model: results/experiment1/dialogue/t5/gen4
tags:
- generated_from_trainer
datasets:
- dialogue
model-index:
- name: gen4
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# gen4

This model is a fine-tuned version of [results/experiment1/dialogue/t5/gen4](https://huggingface.co/results/experiment1/dialogue/t5/gen4) on the dialogue dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 4
- eval_batch_size: 4
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 3.0

### Framework versions

- Transformers 4.36.0.dev0
- Pytorch 2.1.0+cu121
- Datasets 2.14.6
- Tokenizers 0.14.1
