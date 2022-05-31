import json
import torch
import random
import argparse
import logging
from collections import defaultdict
from utils_data import infer_sentence_breaks
from sklearn.model_selection import KFold

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def break_sentences(text):
    sent_locs = list(infer_sentence_breaks(text))
    return [text[st:ed].strip() for (st,ed) in sent_locs]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bioasq_version', type=str, default='9')
    args = parser.parse_args()

    #########################################################################################################
    ########## transform the training data of BioASQ into the cross-validation data of ReQA BioASQ ##########
    #########################################################################################################

    # load questions and corresponding answers from the training dataset of BioASQ
    version = args.bioasq_version
    with open(f'data/BioASQ/{version}b/training{version}b.json') as data_file:
        all_items = json.load(data_file)['questions']
    question2answers = defaultdict(set)
    for item in all_items:
        question = item['body']
        for a in item['ideal_answer']:
            question2answers[question].add(a)
    question2answers = [(k, tuple(v)) for k, v in question2answers.items()]
    logger.info(f"number of questions: {len(question2answers)}")

    # extract QA pairs
    all_questions = []
    all_answers = []
    all_question_ids = []
    for i, item in enumerate(question2answers):
        question = item[0]
        for a in item[1]:
            all_questions += [question]
            all_answers += [a]
            all_question_ids += [i]
    logger.info(f"[training dataset] number of question-answer pairs: {len(all_questions)}")

    # split QA pairs into 5 folds and save their indices
    all_idx = list(range(len(question2answers)))
    kf = KFold(n_splits=5, shuffle=True, random_state=42).split(all_idx)
    kf = [(train_ids.tolist(), dev_ids.tolist()) for train_ids, dev_ids in kf]
    torch.save(kf, f'data/BioASQ/{version}b/kf.pt')

    # save the data used for cross-validation
    cv_data = {
        'questions': all_questions,
        'answers': all_answers,
        'question_ids': all_question_ids,
    }
    with open(f'data/BioASQ/{version}b/cv_data.json', 'w', encoding='utf-8') as f:
        json.dump(cv_data, f)

    #########################################################################################
    ########## transform the test data of BioASQ into the test data of ReQA BioASQ ##########
    #########################################################################################

    # load the test data batches
    all_test_items = []
    for batch_id in range(1, 6):
        with open(f'data/BioASQ/{version}b/{version}B{batch_id}_golden.json') as data_file:
            all_test_items += json.load(data_file)['questions']
    
    # load the sentences of abstracts related to the question
    with open(f'data/BioASQ/{version}b/abstract_sents.json', 'r', encoding='utf-8') as f:
        abstract_sents = json.load(f)
    
    # extract QA pairs and answer candidates
    test_question2answers = defaultdict(set)
    test_candidates = []
    for item in all_test_items:
        question = item['body']
        for a in item['ideal_answer']:
            test_question2answers[question].add(a)
            test_candidates += [a]
            
        for s in item['snippets']:
            if s['beginSection'] == 'abstract' and s['document'] in abstract_sents:
                a = s['text'].strip()
                sents = break_sentences(a)
                if len(sents) > 1:
                    continue
                flag = False
                for sent in abstract_sents[s['document']]:
                    if a in sent:
                        a = sent
                        flag = True
                        break
                if flag:
                    for ds in abstract_sents[s['document']]:
                        if ds != a:
                            test_candidates += [ds]
    
    # record the ground truth answers to each question
    test_questions = []
    test_ground_truths = []
    for q in test_question2answers.keys():
        test_questions.append(q)

    test_candidates = list(set(test_candidates))

    for q in test_questions:
        q_answers = list(test_question2answers[q])
        answer_ids = []
        for a in q_answers:
            answer_ids.append(test_candidates.index(a))
        test_ground_truths.append(answer_ids)
    logger.info(f"[test dataset] number of questions: {len(test_questions)}, number of candidates: {len(test_candidates)}")

    # save test dataset
    test_data = {
        'questions': test_questions,
        'candidates': test_candidates,
        'ground_truths': test_ground_truths
    }
    with open(f'data/BioASQ/{version}b/test_data.json', 'w', encoding='utf-8') as f:
        json.dump(test_data, f)