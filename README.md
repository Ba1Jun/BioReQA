# Biomedical Retrieval Question Answering
This repository provides the PyTorch implementation of our paper "Improving Biomedical Retrieval Question Answering with Transfer Learning and Post-whitening".

## Installation
```bash
# Install huggingface transformers
pip install transformers==3.0.0
```
Note that you should also install `torch` (see [download instruction](https://pytorch.org/)) to use `transformers`.

## Dataset
Our ReQA BioASQ datasets are built on the 6th, 7th, 8th and 9th BioASQ datasets which can be downloaded from the [official website](http://www.bioasq.org/) after registering for the challenge.

For details on the datasets, please see **An overview of the BIOASQ large-scale biomedical semantic indexing and question answering competition (Tsatsaronis et al. 2015)**.


## Models
We use the following version of BioBERT in PyTorch from Transformers which achieves the best performance in our experiments, while the other pre-trained language models can also be used in our framework.
* [`dmis-lab/biobert-base-cased-v1.1`](https://huggingface.co/dmis-lab/biobert-base-cased-v1.1): BioBERT-Base v1.1 (+ PubMed 1M)


## Example

We show the running cases of the baseline model (Dual-BioBERT) and the proposed RBAR-BioBERT on ReQA BioASQ 9 dataset, the other experiments can also be reproduced by using the hyper-parameters provided in the paper.

### Transforming BioASQ dataset into ReQA BioASQ dataset

```bash
# Train the teacher model first

```

### Dual-BioBERT
#### 1. Fine-tuning Dual-BioBERT on ReQA BioASQ dataset
```bash
# Train the teacher model first

```

### RBAR-BioBERT
#### 1. Pre-training on NLI datasets

```bash
# Train the teacher model first

```

#### 2. Fine-tuning on ReQA BioASQ
```bash

```

## License and Disclaimer
Please see the LICENSE file for details. Downloading data indicates your acceptance of our disclaimer.


## Contact
For help or issues using RBAR framework, please create an issue.
