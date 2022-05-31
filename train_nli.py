'''
This script handling the training process.
'''
import warnings
warnings.filterwarnings("ignore")

import os
import argparse
import math
import time
import logging
import json
import random
from tqdm import tqdm
from collections import defaultdict

import numpy as np
import torch

from tensorboardX import SummaryWriter
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer, BertTokenizer



logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def train_epoch(model, train_data_loader, optimizer, scheduler, epoch_i, args):

    model.train()
    total_tr_loss = 0.0
    total_train_batch = 0
    total_acc = 0.0
    start = time.time()

    for step, batch in enumerate(tqdm(train_data_loader, desc='  -(Training)', leave=False)):

        # forward
        tr_loss, tr_acc = model(**batch)
        
        # backward
        tr_loss.backward()

        # record
        total_acc += tr_acc
        total_tr_loss += tr_loss.item()
        total_train_batch += 1

        # update parameters
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        model.zero_grad()
    
    logger.info('[Epoch{epoch: d}] - (Train) loss ={train_loss: 8.5f}, acc ={acc: 3.2f} %, '\
                'elapse ={elapse: 3.2f} min'.format(epoch=epoch_i, 
                                                    train_loss=total_tr_loss / total_train_batch, 
                                                    acc=100 * total_acc / total_train_batch,
                                                    elapse=(time.time()-start)/60))


def dev_eval(model, dev_data_loader, epoch_i, args):
    ''' Epoch operation in evaluation phase '''

    model.eval()

    premise_embeddings = []
    hypothesis_embeddings = []
    # neg_embeddings = []

    with torch.no_grad():
        for batch in tqdm(dev_data_loader, desc='  -(Validation)', leave=False):
            batch['inference'] = True
            # forward
            premise_embedding, hypothesis_embedding = model(**batch)
            premise_embeddings.append(premise_embedding.cpu().numpy())
            hypothesis_embeddings.append(hypothesis_embedding.cpu().numpy())

    premise_embeddings = np.concatenate(premise_embeddings, 0)
    hypothesis_embeddings = np.concatenate(hypothesis_embeddings, 0)

    all_predict_logits = np.matmul(premise_embeddings, hypothesis_embeddings.T)
    all_ground_truth = [i for i in range(all_predict_logits.shape[0])]

    rankat = [1, 5, 10]
    r_counts = defaultdict(float)
    for rank in rankat:
        r_counts[rank] = 0
    r_rank = 0

    for num in range(len(all_ground_truth)):
        pred = np.argsort(-all_predict_logits[num]).tolist()
        for rank in rankat:
            if all_ground_truth[num] in pred[:rank]:
                r_counts[rank] += 1
    
        for idx, p in enumerate(pred):
            if p == all_ground_truth[num]:
                r_rank += 1/(idx+1)
                break

    mrr = np.round(r_rank/len(all_ground_truth), 4)
    r_at_k = [np.round(v/len(all_ground_truth), 4) for k, v in sorted(r_counts.items(), key=lambda item: item[0])]

    logger.info('[Epoch{epoch: d}] - (Dev  ) mrr ={mrr: 3.2f} %, r1 ={r1: 3.2f} %,'\
            ' r5 ={r5: 3.2f} %, r10 ={r10: 3.2f} %'.format(epoch=epoch_i, 
                                                           mrr=mrr*100, 
                                                           r1=r_at_k[0]*100, 
                                                           r5=r_at_k[1]*100, 
                                                           r10=r_at_k[2]*100))

    return mrr*100, r_at_k[0]*100, r_at_k[1]*100, r_at_k[2]*100


def run(model, train_data_loader, dev_data_loader, args):
    args.num_training_steps = int(args.num_train_instances / args.batch_size * args.epoch)
    logger.info("batch size:{}".format(args.batch_size))
    logger.info("total train_steps:{}".format(args.num_training_steps))
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(args.warmup_proportion * args.num_training_steps), 
        num_training_steps=args.num_training_steps
    )
    
    best_metrics = 0
    best_epoch = 0

    for epoch_i in range(args.epoch):
        logger.info('[Epoch {}]'.format(epoch_i))

        train_epoch(model, train_data_loader, optimizer, scheduler, epoch_i, args)

        dev_mrr, dev_r1, dev_r5, dev_r10 = dev_eval(model, dev_data_loader, epoch_i, args)
        current_metrics = dev_r1

        if args.save_model:
            model_name = args.save_model + '/model.pt'
            if not os.path.exists(args.save_model):
                os.makedirs(args.save_model)
            if current_metrics >= best_metrics:
                best_epoch = epoch_i
                best_metrics = current_metrics
                model_state_dict = model.state_dict()
                torch.save(model_state_dict, model_name)
                logger.info('  - [Info] The checkpoint file has been updated.')
    logger.info(f'Got best test performance on epoch{best_epoch}')
    logger.info('\n')


def prepare_dataloaders(args):
    from utils_data import NliDataset as Dataset
    # initialize datasets
    train_dataset = Dataset(args, split='train')
    dev_dataset = Dataset(args, split='dev')
    
    args.num_train_instances = train_dataset.__len__()
    logger.info(f"train data size: {train_dataset.__len__()}")
    logger.info(f"dev data size: {dev_dataset.__len__()}")

    # train dataset

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=train_dataset.collate_fn)

    dev_data_loader = torch.utils.data.DataLoader(
        dev_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=dev_dataset.collate_fn)
    

    return train_data_loader, dev_data_loader


def prepare_model(args):
    from models.dual_encoder import RankModel
    model = RankModel(args)
    model.to(args.device)
    return model



def main():
    ''' Main function '''
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--dataset', type=str, default='nli')
    parser.add_argument('--max_premise_len', type=int)
    parser.add_argument('--max_hypothesis_len', type=int)
    parser.add_argument("--overwrite_cache", action="store_true")
    # training
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--use_neg", action="store_true")
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--warmup_proportion', type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument('--seed', type=int)
    # model
    parser.add_argument('--encoder_type', type=str)
    parser.add_argument('--plm_path', type=str)
    parser.add_argument('--pooler_type', type=str)
    parser.add_argument('--load_model', default='')
    parser.add_argument('--save_model', default=None)
    parser.add_argument('--temperature', type=float)

    args = parser.parse_args()

    logger.info(args)

    # random seed
    set_seed(args)
    
    # set device
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # loading dataset
    train_data_loader, dev_data_loader = prepare_dataloaders(args)

    # preparing model
    model = prepare_model(args)

    # running
    run(model, train_data_loader, dev_data_loader, args)


if __name__ == '__main__':
    main()
