# Biomedical Retrieval Question Answering
This repository provides the PyTorch implementation of our paper "Improving Biomedical ReQA with Consistent NLI-Transfer and Post-Whitening".

## Installation
```bash
# Install huggingface transformers and pytorch
transformers==4.2.1
torch==1.6.0
```

## Dataset
The proposed RBAR framework involves the consistent NLI-transfer and the consistent post-whitening.

For the pre-training on NLI task, we use the combined datasets of SNLI and MultiNLI, which can be downloaded from [AllNLI.tsv.gz](https://sbert.net/datasets/AllNLI.tsv.gz). The downloaded data package should be unzip to "./data/NLI/AllNLI.tsv".

For the training on biomedical ReQA task, we build ReQA BioASQ datasets from the 6th, 7th, 8th and 9th BioASQ datasets. The original BioASQ datasets can be downloaded from the [official website](http://www.bioasq.org/) after registering for the challenge. For the ReQA BioASQ 9b dataset, for example, the training dataset and test dataset batches of BioASQ 9 should be downloaded to "./data/BioASQ/9b/":
```bash
# path: ./data/BioASQ/9b
9B1_golden.json     # the 1st batch of test dataset
9B1_abstract.json   # the json file saving the sentences of abstracts related to the test questions, which is used to build the answer candidates for the test dataset of ReQA BioASQ
training9b.json     # the training dataset of BioASQ 9b
```

For details on the BioASQ datasets, please see **An overview of the BIOASQ large-scale biomedical semantic indexing and question answering competition (Tsatsaronis et al. 2015)**.


## Models
We use the following version of BioBERT in PyTorch from Transformers which achieves the best performance in our experiments, while the other pre-trained language models can also be used in our framework.
* [`dmis-lab/biobert-base-cased-v1.1`](https://huggingface.co/dmis-lab/biobert-base-cased-v1.1): BioBERT-Base v1.1 (+ PubMed 1M)


## Example

We show the running cases of the baseline model (Dual-BioBERT) and the proposed RBAR-BioBERT on ReQA BioASQ 9b dataset, the other experiments can also be reproduced by using the hyper-parameters provided in the paper.

### Transforming BioASQ dataset into ReQA BioASQ dataset
First of all, run the script "run_data_processing.sh".
```bash
# Transform the BioASQ 9b dataset into ReQA BioASQ 9b.
python3 reqa_bioasq_data_processing.py --dataset 9b
```

### Dual-BioBERT
#### 1. Fine-tuning Dual-BioBERT on ReQA BioASQ dataset
run the script "run_reqa_baseline.sh".
```bash
python3 train_reqa.py \
    --seed 12345 \
    --do_train True \
    --do_test True \
    --dev_metric p1 \
    --dataset 6b \
    --max_question_len 24 \
    --max_answer_len 168 \
    --epoch 10 \
    --batch_size 32 \
    --model_type dual_encoder \
    --encoder_type biobert \
    --plm_path /workspace/baijun/models/english/biobert-base-cased-v1.1 \ # your path to the pre-trained model parameters of bioert
    --pooler_type mean \
    --matching_func cos \
    --whitening False \
    --temperature 0.001 \
    --learning_rate 5e-5 \
    --save_model_path output/6b/biobert_baseline/ \
    --rm_saved_model True \
    --save_results True \
```

### RBAR-BioBERT
#### 1. Pre-training on NLI datasets
run the script "run_nli_ranking.sh".
```bash
python3 train_nli_ranking.py \
  --max_premise_len 64 \
  --max_hypothesis_len 32 \
  --epoch 3 \
  --batch_size 32 \
  --model_type dual_encoder_wot \
  --encoder_type biobert \
  --plm_path /workspace/baijun/models/english/biobert-base-cased-v1.1 \ # your path to the pre-trained model parameters of bioert
  --matching_func cos \
  --pooler_type mean \
  --temperature 0.05 \
  --save_model_path output/nli/biobert_ranking \
  --learning_rate 2e-5 \
  --warmup_proportion 0.1 \
  --seed 12345
```

#### 2. Fine-tuning on ReQA BioASQ
run the script "run_reqa_rbar.sh".
```bash
python3 train_reqa.py \
    --seed 12345 \
    --do_train True \
    --do_test True \
    --dev_metric p1 \
    --dataset 6b \
    --max_question_len 24 \
    --max_answer_len 168 \
    --epoch 10 \
    --batch_size 32 \
    --model_type dual_encoder_wot \
    --encoder_type biobert \
    --plm_path /workspace/baijun/models/english/biobert-base-cased-v1.1 \ # your path to the pre-trained model parameters of bioert
    --pooler_type mean \
    --matching_func cos \
    --whitening True \
    --temperature 0.05 \
    --learning_rate 5e-5 \
    --save_model_path output/6b/biobert_rbar/ \
    --load_model_path output/nli/biobert_ranking/model.pt \
    --rm_saved_model True \
    --save_results True \
```

## License and Disclaimer
Please see the LICENSE file for details. Downloading data indicates your acceptance of our disclaimer.


## Contact
For help or issues using RBAR framework, please create an issue.
