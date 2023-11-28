---
base_model: results/experiment1/dialogue/t5/gen2
tags:
- generated_from_trainer
datasets:
- dialogue
metrics:
- rouge
model-index:
- name: gen2
  results:
  - task:
      name: Summarization
      type: summarization
    dataset:
      name: dialogue
      type: dialogue
    metrics:
    - name: Rouge1
      type: rouge
      value: 99.8694
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# gen2

This model is a fine-tuned version of [results/experiment1/dialogue/t5/gen2](https://huggingface.co/results/experiment1/dialogue/t5/gen2) on the dialogue dataset.
It achieves the following results on the evaluation set:
- Loss: 0.0444
- Rouge1: 99.8694
- Rouge2: 99.8205
- Rougel: 99.8586
- Rougelsum: 99.8669
- Gen Len: 26.2256

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

### Training results



### Framework versions

- Transformers 4.36.0.dev0
- Pytorch 2.1.0+cu121
- Datasets 2.14.6
- Tokenizers 0.14.1
