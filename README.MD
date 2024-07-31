# Fine-tuning the Florence-2 using multiple GPUs

This repository is intended for understanding the process of fine-tuning the Florence-2 model. It retains only the essential functionality for multi-GPU training, stripping away most non-critical features. This repository heavily references the [Hugging Face blog on fine-tuning Florence-2](https://huggingface.co/blog/finetune-florence2), with the multi-GPU training code expanded and all bugs manually removed.

## Prerequisites

This repo has been tested under the following dependencies:

- PyTorch 2.3.1
- transformers 4.42.4
- datasets 2.20.0
- tqdm 4.66.4
- florence-2 model @ commited on Jun 30, 2024

## Usage

Just run

```
python florence-2-finetune-distributed.py
```