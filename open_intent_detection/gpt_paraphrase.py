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

ori_examples, labeled_examples, unlabeled_examples = get_examples(args, data.get_attrs(), mode='train')
eval_examples = get_examples(args, data.get_attrs(), mode='eval')
test_examples = get_examples(args, data.get_attrs(), mode='test')

paraphrase(labeled_examples, num_paraphrases = 10, dump_file = "%s_train_paraphrases.csv"%args.dataset)
paraphrase(eval_examples, num_paraphrases = 10, dump_file = "%s_eval_paraphrases.csv"%args.dataset)
paraphrase(test_examples, num_paraphrases = 10, dump_file = "%s_test_paraphrases.csv"%args.dataset)

