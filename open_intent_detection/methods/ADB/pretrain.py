import torch
import torch.nn.functional as F
import numpy as np
import os
import copy
import logging
from torch import nn
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from tqdm import trange, tqdm
from losses import loss_map
from utils.functions import save_model, restore_model, centroids_cal
from dataloaders.bert_loader import DatasetProcessor, convert_examples_to_features, get_loader, get_examples
from utils.openai import paraphrase
import csv
class PretrainManager:
    
    def __init__(self, args, data, model, logger_name = 'Detection'):

        self.logger = logging.getLogger(logger_name)

        self.set_model_optimizer(args, data, model)
        
        self.train_dataloader = data.dataloader.train_labeled_loader
        self.eval_dataloader = data.dataloader.eval_loader
        self.test_dataloader = data.dataloader.test_loader

        self.loss_fct = loss_map[args.loss_fct]  
        self.centroids = None
        self.best_eval_score = None

        if args.pretrain or (not os.path.exists(args.model_output_dir)):
            self.logger.info('Pre-training Begin...')

            if args.backbone == 'bert_disaware':
                self.train_disaware(args, data)
            elif args.backbone == 'bert_disaware_boost':
                self.train_disaware_boost(args, data)
            elif args.backbone == 'bert_boost':
                self.train_boost(args, data)
            else:
                self.train_plain(args, data)

            self.logger.info('Pre-training finished...')
                
        else:
            self.model = restore_model(self.model, args.model_output_dir)

    def set_model_optimizer(self, args, data, model):
    
        self.model = model.set_model(args, 'bert')  
        self.optimizer, self.scheduler = model.set_optimizer(self.model, data.dataloader.num_train_examples, args.train_batch_size, \
                args.num_train_epochs, args.lr, args.warmup_proportion)
        self.device = model.device

    def train_plain(self, args, data):
        
        wait = 0
        best_model = None
        best_eval_score = 0
        
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            
            for step, batch in enumerate(tqdm(self.train_dataloader, desc="Iteration")):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, _ = batch
                with torch.set_grad_enabled(True):
                    loss = self.model(input_ids, segment_ids, input_mask, label_ids, mode = "train", loss_fct = self.loss_fct)
                    self.optimizer.zero_grad()

                    loss.backward()

                    self.optimizer.step()
                    self.scheduler.step()

                    tr_loss += loss.item()
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1
            
            loss = tr_loss / nb_tr_steps
            
            y_true, y_pred = self.get_outputs(args, data, mode = 'eval')
            eval_score = round(accuracy_score(y_true, y_pred) * 100, 2)

            eval_results = {
                'train_loss': loss,
                'eval_score': eval_score,
                'best_eval_score':best_eval_score,
            }
            self.logger.info("***** Epoch: %s: Eval results *****", str(epoch + 1))
            for key in sorted(eval_results.keys()):
                self.logger.info("  %s = %s", key, str(eval_results[key]))
            
            if eval_score > best_eval_score:
                
                best_model = copy.deepcopy(self.model)
                wait = 0
                best_eval_score = eval_score

            elif eval_score > 0:

                wait += 1
                if wait >= args.wait_patient:
                    break

        self.model = best_model

        if args.save_model:
            self.logger.info('Trained models are saved in %s', args.model_output_dir)
            save_model(self.model, args.model_output_dir)

    def train_boost(self, args, data):
        '''
        import traceback
        for line in traceback.format_stack():
            print(line.strip())
        '''

        wait = 0
        best_model = None
        best_eval_score = 0
        
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            
            for step, batch in enumerate(tqdm(self.train_dataloader, desc="Iteration")):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, _ = batch
                with torch.set_grad_enabled(True):
                    loss = self.model(input_ids, segment_ids, input_mask, label_ids, mode = "train", loss_fct = self.loss_fct)
                    self.optimizer.zero_grad()

                    loss.backward()

                    self.optimizer.step()
                    self.scheduler.step()

                    tr_loss += loss.item()
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1
            
            loss = tr_loss / nb_tr_steps
            
            if best_eval_score > args.boost_start_score or wait > 0: 
                self.boost(args, data)
            
            y_true, y_pred = self.get_outputs(args, data, mode = 'eval')
            eval_score = round(accuracy_score(y_true, y_pred) * 100, 2)

            eval_results = {
                'train_loss': loss,
                'eval_score': eval_score,
                'best_eval_score':best_eval_score,
            }
            self.logger.info("***** Epoch: %s: Eval results *****", str(epoch + 1))
            for key in sorted(eval_results.keys()):
                self.logger.info("  %s = %s", key, str(eval_results[key]))
            
            if eval_score > best_eval_score:
                
                best_model = copy.deepcopy(self.model)
                wait = 0
                best_eval_score = eval_score

            elif eval_score > 0:

                wait += 1
                if wait >= args.wait_patient:
                    break

        self.model = best_model

        if args.save_model:
            self.logger.info('Trained models are saved in %s', args.model_output_dir)
            save_model(self.model, args.model_output_dir)

    def train_disaware(self, args, data):

        wait = 0
        best_model = None
        best_centroids = None
        best_eval_score = 0
        args.device = self.device
        
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            self.centroids = centroids_cal(self.model, args, data, self.train_dataloader, self.device)
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            for step, batch in enumerate(tqdm(self.train_dataloader, desc="Iteration")):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, _ = batch

                with torch.set_grad_enabled(True):
                    
                    loss = self.model(input_ids, segment_ids, input_mask, label_ids, mode = "train", loss_fct = self.loss_fct, centroids = self.centroids)

                    self.optimizer.zero_grad()

                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    
                    tr_loss += loss.item()
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1
            
            loss = tr_loss / nb_tr_steps

            y_true, y_pred = self.get_outputs(args, data, mode = 'eval')
            eval_score = round(f1_score(y_true, y_pred, average = 'macro') * 100, 2)

            eval_results = {
                'train_loss': loss,
                'eval_score': eval_score,
                'best_eval_score':best_eval_score,
            }
            self.logger.info("***** Epoch: %s: Eval results *****", str(epoch + 1))
            for key in sorted(eval_results.keys()):
                self.logger.info("  %s = %s", key, str(eval_results[key]))
            
            if eval_score > best_eval_score:
                
                best_model = copy.deepcopy(self.model)
                best_centroids = copy.copy(self.centroids)
                wait = 0
                best_eval_score = eval_score

            elif eval_score > 0:

                wait += 1
                if wait >= args.wait_patient:
                    break

        self.model = best_model
        self.centroids = best_centroids
        self.best_eval_score = best_eval_score

        if args.save_model:
            self.logger.info('Trained models are saved in %s', args.model_output_dir)
            save_model(self.model, args.model_output_dir)       
    
    def train_disaware_boost(self, args, data):

        wait = 0
        best_model = None
        best_centroids = None
        best_eval_score = 0
        args.device = self.device
        
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            self.centroids = centroids_cal(self.model, args, data, self.train_dataloader, self.device)
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            for step, batch in enumerate(tqdm(self.train_dataloader, desc="Iteration")):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, _ = batch

                with torch.set_grad_enabled(True):
                    
                    loss = self.model(input_ids, segment_ids, input_mask, label_ids, mode = "train", loss_fct = self.loss_fct, centroids = self.centroids)

                    self.optimizer.zero_grad()

                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    
                    tr_loss += loss.item()
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1
            
            loss = tr_loss / nb_tr_steps
            
            if best_eval_score > args.boost_start_score or wait > 0: 
                self.boost(args, data)

            y_true, y_pred = self.get_outputs(args, data, mode = 'eval')
            eval_score = round(f1_score(y_true, y_pred, average = 'macro') * 100, 2)

            eval_results = {
                'train_loss': loss,
                'eval_score': eval_score,
                'best_eval_score':best_eval_score,
            }
            self.logger.info("***** Epoch: %s: Eval results *****", str(epoch + 1))
            for key in sorted(eval_results.keys()):
                self.logger.info("  %s = %s", key, str(eval_results[key]))
            
            if eval_score > best_eval_score:
                
                best_model = copy.deepcopy(self.model)
                best_centroids = copy.copy(self.centroids)
                wait = 0
                best_eval_score = eval_score

            elif eval_score > 0:

                wait += 1
                if wait >= args.wait_patient:
                    break

        self.model = best_model
        self.centroids = best_centroids
        self.best_eval_score = best_eval_score

        if args.save_model:
            self.logger.info('Trained models are saved in %s', args.model_output_dir)
            save_model(self.model, args.model_output_dir)         
            
    def boost(self, args, data):
        print("---begin boost---")
        
        ori_examples, labeled_examples, unlabeled_examples = get_examples(args, data.get_attrs(), mode='train')
        method = args.boost_method
        if method.startswith("WP"):
        
            train_dataloader = self.train_dataloader
            self.model.eval()
            
            total_labels = torch.empty(0,dtype=torch.long).to(self.device)
            total_preds = torch.empty(0,dtype=torch.long).to(self.device)
            
            total_features = torch.empty((0,args.feat_dim)).to(self.device)
            total_logits = torch.empty((0, data.num_labels)).to(self.device)
            
            total_guids = torch.empty(0,dtype=torch.long).to(self.device)
            
            total_input_ids = torch.empty((0,args.max_seq_length)).to(self.device)
            
            for batch in tqdm(train_dataloader, desc="Iteration"):
    
                batch = tuple(t.to(self.device) for t in batch)
                
                input_ids, input_mask, segment_ids, label_ids, guids = batch
                with torch.set_grad_enabled(False):
                    if args.backbone == 'bert_disaware_boost':
                        pooled_output, logits = self.model(input_ids, segment_ids, input_mask, centroids = self.centroids, labels = label_ids, mode = 'eval')        
                    elif args.backbone == 'bert_boost':
                        pooled_output, logits = self.model(input_ids, segment_ids, input_mask, labels = label_ids, mode = 'eval')
                    else:
                        raise NotImplementedError()
                    total_labels = torch.cat((total_labels,label_ids))
                    total_features = torch.cat((total_features, pooled_output))
                    total_logits = torch.cat((total_logits, logits))
                    
                    total_guids = torch.cat((total_guids,guids))
                    
                    total_input_ids = torch.cat((total_input_ids, input_ids))
                    
            total_probs = F.softmax(total_logits.detach(), dim=1)
            total_maxprobs, total_preds = total_probs.max(dim = 1)
    
            y_pred = total_preds.cpu().numpy()
            y_true = total_labels.cpu().numpy()
            
            prediction_map = {}
    
            #threshold_prob = np.percentile(total_maxprobs[y_pred != y_true].cpu().numpy(), 50)
            
            tagged_guids = []
            for i in range(len(y_pred)):
                if y_true[i] != y_pred[i]: # and total_maxprobs[i]>threshold_prob:
                    prediction_map[total_guids[i].item()] = (y_pred[i], y_true[i])
                    tagged_guids.append(total_guids[i].item())
            
            print("len(tagged_guids):", len(tagged_guids))
            
            example_map = dict([(labeled_example.guid, labeled_example) for labeled_example in labeled_examples])
            
            para_examples = [example_map[tagged_guid] for tagged_guid in tagged_guids]
            
            num_paraphrases = int(method.split('-')[1])
            
        elif method.startswith("F"):
            para_examples = labeled_examples
            num_paraphrases = int(method.split('-')[1])
        else:
            raise NotImplementedError()
        
        if len(para_examples) > 0:
           
            para_examples = paraphrase(para_examples, num_paraphrases = num_paraphrases, cache_file = "%s_train_paraphrases.csv"%args.dataset)
            
            if method == "wrong_prediction":
                with open("%s_train_dump.csv"%args.dataset,"a",newline="\n") as f:
                    csv_writer=csv.writer(f)
                    csv_writer.writerows([("GUID", "Original Example", "ChatGPT's Paraphrase", "Ground Truth Label", "Prediction Label")])
                    csv_writer.writerows([(para_example.guid, para_example.text_b, para_example.text_a, para_example.label, data.label_list[prediction_map[para_example.guid][0]]) for para_example in para_examples])  
                
            
            para_dataloader = get_loader(para_examples, args, data.label_list, mode = 'train_labeled', sampler_mode = 'random')
            
            self.model.train()
    
            for step, batch in enumerate(tqdm(para_dataloader, desc="Iteration")):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, _ = batch
    
                with torch.set_grad_enabled(True):
                    if args.backbone == 'bert_disaware_boost':
                        loss = self.model(input_ids, segment_ids, input_mask, label_ids, mode = "train", loss_fct = self.loss_fct, centroids = self.centroids)      
                    elif args.backbone == 'bert_boost':
                        loss = self.model(input_ids, segment_ids, input_mask, label_ids, mode = "train", loss_fct = self.loss_fct)
                    else:
                        raise NotImplementedError()
                    self.optimizer.zero_grad()
    
                    loss.backward()
                    self.optimizer.step()
                    #self.scheduler.step()
            
        print("---end boost---")

    def get_outputs(self, args, data, mode = 'eval', get_feats = False):
        
        if mode == 'eval':
            dataloader = self.eval_dataloader
        elif mode == 'test':
            dataloader = self.test_dataloader

        self.model.eval()

        total_labels = torch.empty(0,dtype=torch.long).to(self.device)
        total_preds = torch.empty(0,dtype=torch.long).to(self.device)
        
        total_features = torch.empty((0,args.feat_dim)).to(self.device)
        total_logits = torch.empty((0, data.num_labels)).to(self.device)
        
        for batch in tqdm(dataloader, desc="Iteration"):

            batch = tuple(t.to(self.device) for t in batch)
            
            input_ids, input_mask, segment_ids, label_ids, _ = batch
            with torch.set_grad_enabled(False):

                if args.backbone == 'bert_disaware' or args.backbone == 'bert_disaware_boost':
                    pooled_output, logits = self.model(input_ids, segment_ids, input_mask, centroids = self.centroids, labels = label_ids, mode = mode)
                else:    
                    pooled_output, logits = self.model(input_ids, segment_ids, input_mask, mode = mode)

                total_labels = torch.cat((total_labels,label_ids))
                total_features = torch.cat((total_features, pooled_output))
                total_logits = torch.cat((total_logits, logits))

        if get_feats:  
            feats = total_features.cpu().numpy()
            return feats

        else:

            total_probs = F.softmax(total_logits.detach(), dim=1)
            total_maxprobs, total_preds = total_probs.max(dim = 1)

            y_pred = total_preds.cpu().numpy()
            y_true = total_labels.cpu().numpy()

            return y_true, y_pred
