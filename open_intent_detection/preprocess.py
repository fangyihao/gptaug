'''
Created on Apr. 18, 2023

@author: Yihao Fang
'''
import torch
import torch.nn.functional as F
import numpy as np
import os
import logging
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from tqdm import trange, tqdm
from utils.functions import save_model, euclidean_metric
from utils.metrics import F_measure
from dataloaders.bert_loader import DatasetProcessor, convert_examples_to_features, get_loader, get_examples
import logging
import argparse
import sys
import os
import datetime
from configs.base import ParamManager
from dataloaders.base import DataManager
from utils.openai import paraphrase
from rouge import Rouge
from collections import OrderedDict
from copy import deepcopy
import pandas as pd
import random
from ordered_set import OrderedSet
rouge = Rouge()


def parse_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument('--type', type=str, default='open_intent_detection', help="Type style.")

    parser.add_argument('--logger_name', type=str, default='Detection', help="Logger name for open intent detection.")

    parser.add_argument('--log_dir', type=str, default='logs', help="Logger directory.")

    parser.add_argument("--dataset", default='banking', type=str, help="The name of the dataset to train selected")

    parser.add_argument("--known_cls_ratio", default=0.75, type=float, help="The number of known classes")
    
    parser.add_argument("--labeled_ratio", default=1.0, type=float, help="The ratio of labeled samples in the training set")
    
    parser.add_argument("--method", type=str, default='ADB', help="which method to use")

    parser.add_argument("--train", action="store_true", help="Whether to train the model")

    parser.add_argument("--pretrain", action="store_true", help="Whether to pre-train the model")

    parser.add_argument("--save_model", action="store_true", help="save trained-model for open intent detection")

    parser.add_argument("--backbone", type=str, default='bert', help="which backbone to use")

    parser.add_argument("--config_file_name", type=str, default='ADB.py', help = "The name of the config file.")

    parser.add_argument('--seed', type=int, default=0, help="random seed for initialization")

    parser.add_argument("--gpu_id", type=str, default='0', help="Select the GPU id")

    parser.add_argument("--pipe_results_path", type=str, default='pipe_results', help="the path to save results of pipeline methods")
    
    parser.add_argument("--data_dir", default = sys.path[0]+'/../data', type=str,
                        help="The input data dir. Should contain the .csv files (or other data files) for the task.")

    parser.add_argument("--output_dir", default= '/home/yfang/workspace/TEXTOIR/output', type=str, 
                        help="The output directory where all train data will be written.") 

    parser.add_argument("--model_dir", default='models', type=str, 
                        help="The output directory where the model predictions and checkpoints will be written.") 

    parser.add_argument("--load_pretrained_method", default=None, type=str, 
                        help="The output directory where the model predictions and checkpoints will be written.") 

    parser.add_argument("--result_dir", type=str, default = 'results', help="The path to save results")

    parser.add_argument("--results_file_name", type=str, default = 'results.csv', help="The file name of all the results.")

    parser.add_argument("--save_results", action="store_true", help="save final results for open intent detection")

    parser.add_argument("--loss_fct", default="CrossEntropyLoss", help="The loss function for training.")
    
    parser.add_argument("--exit_target_degree", default=0, type=float, help="The maximum degree of the target data nodes to exit")
    
    parser.add_argument("--rouge_threshold", default=0.2, type=float, help="The maximum degree of the target data nodes to exit")

    args = parser.parse_args()

    return args

def set_logger(args):


    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    
    time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    file_name = f"{args.method}_{args.dataset}_{args.backbone}_{args.known_cls_ratio}_{args.labeled_ratio}_{time}.log"
    
    logger = logging.getLogger(args.logger_name)
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(os.path.join(args.log_dir, file_name))
    fh_formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
    fh.setFormatter(fh_formatter)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter('%(name)s - %(message)s')
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    return logger
    
def find_similar_pairs(example_map, src_mode, trgt_mode, rouge_threshold):   
    similar_pairs = []
    for label in example_map[src_mode]:
        per_label_src_examples = example_map[src_mode][label]
        per_label_trgt_examples = example_map[trgt_mode][label]
        for per_label_src_example in per_label_src_examples:
            for per_label_trgt_example in per_label_trgt_examples:
                print("%s example:"%src_mode, per_label_src_example.text_a)
                print("%s example:"%trgt_mode, per_label_trgt_example.text_a)
                rouge_score = rouge.get_scores(per_label_src_example.text_a, per_label_trgt_example.text_a)[0]["rouge-l"]["f"]
                print("rouge score:", rouge_score)
                print("-"*80)
                if rouge_score > rouge_threshold:
                    similar_pairs.append((per_label_src_example, per_label_trgt_example))
    return similar_pairs

def find_pruned_guids(similar_pairs, src_mode, trgt_mode, example_map, exit_target_degree):
    similar_pairs = deepcopy(similar_pairs)
    print("# of %s-%s similar pairs:"%(src_mode, trgt_mode), len(similar_pairs))
    pruned_guid_map = {src_mode:OrderedDict(),trgt_mode:OrderedDict()}
    
    while len(similar_pairs) > 0:
        degree_map={src_mode:OrderedDict(),trgt_mode:OrderedDict()}
        
        labels = OrderedSet()
        for similar_pair in similar_pairs:
            if similar_pair[0].label not in degree_map[src_mode]:
                degree_map[src_mode][similar_pair[0].label]=OrderedDict()
            if similar_pair[0].guid not in degree_map[src_mode][similar_pair[0].label]:
                degree_map[src_mode][similar_pair[0].label][similar_pair[0].guid] = 0
            degree_map[src_mode][similar_pair[0].label][similar_pair[0].guid] += 1
            
            if similar_pair[1].label not in degree_map[trgt_mode]:
                degree_map[trgt_mode][similar_pair[1].label]=OrderedDict()
            if similar_pair[1].guid not in degree_map[trgt_mode][similar_pair[1].label]:
                degree_map[trgt_mode][similar_pair[1].label][similar_pair[1].guid] = 0
            degree_map[trgt_mode][similar_pair[1].label][similar_pair[1].guid] += 1
            
            labels.add(similar_pair[0].label)
            labels.add(similar_pair[1].label)
        
        trgt_degrees = []
        for label in labels:
            src_node = sorted(degree_map[src_mode][label].items(), key=lambda x:x[1])[-1]
            trgt_node = sorted(degree_map[trgt_mode][label].items(), key=lambda x:x[1])[-1]
            
            trgt_degrees.append(trgt_node[1])
            #print(src_node[1], trgt_node[1])
            #if trgt_node[1] <= 5:
            #    break    
            
            #if src_node[1]*(src_size-len(pruned_src_guids)) > trgt_node[1]*(trgt_size-len(pruned_trgt_guids)):
            src_weight = len(example_map[src_mode][label])-(len(pruned_guid_map[src_mode][label]) if label in pruned_guid_map[src_mode] else 0)
            trgt_weight = len(example_map[trgt_mode][label])-(len(pruned_guid_map[trgt_mode][label]) if label in pruned_guid_map[trgt_mode] else 0)
            if src_node[1]*src_weight > trgt_node[1]*trgt_weight:
                if label not in pruned_guid_map[src_mode]:
                    pruned_guid_map[src_mode][label] = []
                pruned_guid_map[src_mode][label].append(src_node[0])
                similar_pairs = [similar_pair for similar_pair in similar_pairs if similar_pair[0].guid != src_node[0]]
            else:
                if label not in pruned_guid_map[trgt_mode]:
                    pruned_guid_map[trgt_mode][label] = []
                pruned_guid_map[trgt_mode][label].append(trgt_node[0])
                similar_pairs = [similar_pair for similar_pair in similar_pairs if similar_pair[1].guid != trgt_node[0]]
                
        if np.mean(trgt_degrees) <= exit_target_degree:
            break
        #print("# of %s-%s similar pairs:"%(src_mode, trgt_mode), len(similar_pairs))
        
    pruned_src_guids = []
    pruned_trgt_guids = []
    for label in pruned_guid_map[src_mode]:
        for guid in pruned_guid_map[src_mode][label]:
            pruned_src_guids.append(guid)
    for label in pruned_guid_map[trgt_mode]:
        for guid in pruned_guid_map[trgt_mode][label]:
            pruned_trgt_guids.append(guid)
        
    return pruned_src_guids, pruned_trgt_guids


def prune(args, data, logger):
    ori_examples, labeled_examples, unlabeled_examples = get_examples(args, data.get_attrs(), mode='train')
    eval_examples = get_examples(args, data.get_attrs(), mode='eval')
    test_examples = get_examples(args, data.get_attrs(), mode='test')
    
    example_map={'train':{},'eval':{},'test':{}}
    for mode, examples in zip(['train', 'eval', 'test'], [labeled_examples, eval_examples, test_examples]):
        for example in examples:
            if example.label not in example_map[mode]:
                example_map[mode][example.label] = []
            example_map[mode][example.label].append(example)
    
    train_test_similar_pairs = find_similar_pairs(example_map, 'train', 'test', args.rouge_threshold)
    
    pruned_train_guids, pruned_test_guids = find_pruned_guids(train_test_similar_pairs, 'train', 'test', example_map, args.exit_target_degree)
    
    for label in example_map['train']:
        example_map['train'][label] = [example for example in example_map['train'][label] if example.guid not in pruned_train_guids]
        
    for label in example_map['test']:
        example_map['test'][label] = [example for example in example_map['test'][label] if example.guid not in pruned_test_guids]
        
                        
    train_eval_similar_pairs = find_similar_pairs(example_map, 'train', 'eval', args.rouge_threshold)
    
    pruned_train_guids_2, pruned_eval_guids = find_pruned_guids(train_eval_similar_pairs, 'train', 'eval', example_map, args.exit_target_degree)
    pruned_train_guids.extend(pruned_train_guids_2)
    
    print("-"*80)
    print("# of train examples to prune:", len(pruned_train_guids))
    print("# of eval examples to prune:", len(pruned_eval_guids))
    print("# of test examples to prune:", len(pruned_test_guids))
    print("-"*80)
    
    pd.set_option('display.max_rows', None)
    
    train_df = pd.read_csv(os.path.join(data.get_attrs()['data_dir'], "train.tsv"), sep="\t")
    print("train dataset before:")
    print(train_df.groupby(['label']).size())
    print(len(train_df.groupby(['label']).size()))
    train_df = pd.DataFrame([(example.text_a, example.label) for example in labeled_examples if example.guid not in pruned_train_guids], columns =['text', 'label'])
    print("train dataset after:")
    print(train_df.groupby(['label']).size())
    print(len(train_df.groupby(['label']).size()))
    train_df.to_csv("../data/%s_cg/train.tsv"%args.dataset, encoding='utf-8', index=False, sep="\t")
    
    eval_df = pd.read_csv(os.path.join(data.get_attrs()['data_dir'], "dev.tsv"), sep="\t")
    print("eval dataset before:")
    print(eval_df.groupby(['label']).size())
    print(len(eval_df.groupby(['label']).size()))
    eval_df = pd.DataFrame([(example.text_a, example.label) for example in eval_examples if example.guid not in pruned_eval_guids], columns =['text', 'label'])
    print("eval dataset after:")
    print(eval_df.groupby(['label']).size())
    print(len(eval_df.groupby(['label']).size()))
    eval_df.to_csv("../data/%s_cg/dev.tsv"%args.dataset, encoding='utf-8', index=False, sep="\t")
    
    test_df = pd.read_csv(os.path.join(data.get_attrs()['data_dir'], "test.tsv"), sep="\t")
    print("test dataset before:")
    print(test_df.groupby(['label']).size())
    print(len(test_df.groupby(['label']).size()))
    test_df = pd.DataFrame([(example.text_a, example.label) for example in test_examples if example.guid not in pruned_test_guids], columns =['text', 'label'])
    print("test dataset after:")
    print(test_df.groupby(['label']).size())
    print(len(test_df.groupby(['label']).size()))
    test_df.to_csv("../data/%s_cg/test.tsv"%args.dataset, encoding='utf-8', index=False, sep="\t")
    
def paraphrase(args, data, logger):
    set_type_map = {'train': 0, 'dev': 1e8, 'test': 2e8}
    for mode in ['train', 'dev', 'test']:
        para_df = pd.read_csv("%s_%s_paraphrases.csv"%(args.dataset, mode), encoding='utf-8')
        para_df.groupby(['GUID']).size().to_csv("%s_%s_paraphrases_stats.csv"%(args.dataset, mode), encoding='utf-8', index=True)
        
        cg_df = pd.read_csv("../data/%s_cg/%s.tsv"%(args.dataset, mode), encoding='utf-8', sep="\t")
        start = int(set_type_map[mode]+1)
        cg_df.insert(0, 'GUID', range(start, start + cg_df.shape[0]))
        para_cg_df = cg_df.merge(para_df[["Original Example", "ChatGPT's Paraphrase", "Label"]], how='left',left_on='text', right_on='Original Example').fillna("")
        para_cg_df = para_cg_df[["GUID", "Original Example", "ChatGPT's Paraphrase", "Label"]]
        para_cg_df.to_csv("%s_cg_%s_paraphrases.csv"%(args.dataset, mode), encoding='utf-8', index=False)
        
        #pd.set_option('display.max_rows', None)
        para_cg_df.groupby(['GUID']).size().to_csv("%s_cg_%s_paraphrases_stats.csv"%(args.dataset, mode), encoding='utf-8', index=True)

def main():   
    sys.path.append('.')
    args = parse_arguments()
    logger = set_logger(args)
    
    param = ParamManager(args)
    args = param.args
    
    logger.debug("="*30+" Params "+"="*30)
    for k in args.keys():
        logger.debug(f"{k}:\t{args[k]}")
    logger.debug("="*30+" End Params "+"="*30)
    
    data = DataManager(args, logger_name = args.logger_name)
    
    #prune(args, data, logger)
    paraphrase(args, data, logger)
    
if __name__ == '__main__':    
    main()


