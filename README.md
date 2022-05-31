# Biomedical Retrieval Question Answering
This repository provides the PyTorch implementation of our paper "Improving Biomedical Retrieval Question Answering with Transfer Learning and Post-whitening".

## Installation
```bash
# Install huggingface transformers
transformers==4.2.1
torch==1.6.0
scikit-learn==0.21.0
```

## Dataset
The proposed RBAR framework involves the pre-training on NLI task and the subsequent training of on biomedical ReQA task.

For the pre-training on NLI, we use the combined datasets of SNLI and MultiNLI, which can be downloaded from [AllNLI.tsv.gz](https://sbert.net/datasets/AllNLI.tsv.gz). The downloaded data package should be unzip to "./data/NLI/AllNLI.tsv".

For the training on biomedical ReQA, we build ReQA BioASQ datasets from the 6th, 7th, 8th and 9th BioASQ datasets. The original BioASQ datasets can be downloaded from the [official website](http://www.bioasq.org/) after registering for the challenge. For the ReQA BioASQ 9 dataset, for example, the training dataset and test dataset batches of BioASQ 9 should be downloaded to "./data/BioASQ/9b/":
```bash
# path: ./data/BioASQ/9b
9B1_golden.json     # the 1st batch of test dataset
9B2_golden.json     # the 2nd batch of test dataset
9B3_golden.json     # the 3rd batch of test dataset
9B4_golden.json     # the 4th batch of test dataset
9B5_golden.json     # the 5th batch of test dataset
abstract_sents.json # the json file saving the sentences of abstracts related to the test questions, which is used to build the answer candidates for the test dataset of ReQA BioASQ
training9b.json     # the training dataset of BioASQ 9
```

For details on the BioASQ datasets, please see **An overview of the BIOASQ large-scale biomedical semantic indexing and question answering competition (Tsatsaronis et al. 2015)**.


## Models
We use the following version of BioBERT in PyTorch from Transformers which achieves the best performance in our experiments, while the other pre-trained language models can also be used in our framework.
* [`dmis-lab/biobert-base-cased-v1.1`](https://huggingface.co/dmis-lab/biobert-base-cased-v1.1): BioBERT-Base v1.1 (+ PubMed 1M)


## Example

We show the running cases of the baseline model (Dual-BioBERT) and the proposed RBAR-BioBERT on ReQA BioASQ 9 dataset, the other experiments can also be reproduced by using the hyper-parameters provided in the paper.

### Transforming BioASQ dataset into ReQA BioASQ dataset

```bash
# Transform the BioASQ 9 dataset into ReQA BioASQ 9.
python3 dataset_process_bioasq.py --bioasq_version 9b
```

### Dual-BioBERT
#### 1. Fine-tuning Dual-BioBERT on ReQA BioASQ dataset
run the script "run_baseline.sh".
```bash
python3 train_reqa.py \
      --do_test \
      --rm_saved_model \
      --dataset 9b \
      --max_question_len 32 \
      --max_answer_len 256 \
      --epoch 10 \
      --batch_size 32 \
      --model_type dual_encoder \
      --encoder_type biobert \
      --plm_path ./models/biobert-base-cased-v1.1 \ # your path to the pre-trained model parameters of bioert
      --pooler_type mean \
      --temperature 0.05 \
      --learning_rate 2e-5 \
      --save_model output/baseline \
```

### RBAR-BioBERT
#### 1. Pre-training on NLI datasets
run the script "run_nli_pre_train.sh".
```bash
python3 train_nli.py \
  --max_premise_len 64 \
  --max_hypothesis_len 32 \
  --epoch 3 \
  --batch_size 32 \
  --encoder_type biobert \
  --plm_path ./models/biobert-base-cased-v1.1 \ # your path to the pre-trained model parameters of bioert
  --pooler_type mean \
  --temperature 0.05 \
  --save_model output/nli/ \
  --learning_rate 2e-5 \
  --warmup_proportion 0.1 \
  --seed 42 \
```

#### 2. Fine-tuning on ReQA BioASQ
run the script "run_rbar.sh".
```bash
python3 train_reqa.py \
      --do_test \
      --rm_saved_model \
      --dataset 9b \
      --max_question_len 32 \
      --max_answer_len 256 \
      --epoch 10 \
      --batch_size 32 \
      --model_type dual_encoder \
      --encoder_type biobert \
      --plm_path ./models/biobert-base-cased-v1.1 \ # your path to the pre-trained model parameters of bioert
      --pooler_type mean \
      --temperature 0.05 \
      --learning_rate 5e-5 \
      --save_model output/rbar \
      --load_model output/nli/model.pt \  # the model parameters pre-trained on NLI dataset
```

## License and Disclaimer
Please see the LICENSE file for details. Downloading data indicates your acceptance of our disclaimer.


## Contact
For help or issues using RBAR framework, please create an issue.
