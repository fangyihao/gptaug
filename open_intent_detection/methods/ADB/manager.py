import torch
import torch.nn.functional as F
import numpy as np
import os
import logging
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from tqdm import trange, tqdm
from .boundary import BoundaryLoss
from losses import loss_map
from utils.functions import save_model, euclidean_metric
from utils.metrics import F_measure
from utils.functions import restore_model, centroids_cal
from .pretrain import PretrainManager
from dataloaders.bert_loader import DatasetProcessor, convert_examples_to_features, get_loader, get_examples
from utils.openai import paraphrase

class ADBManager:
    
    def __init__(self, args, data, model, logger_name = 'Detection'):

        self.logger = logging.getLogger(logger_name)

        pretrain_model = PretrainManager(args, data, model)
        self.model = pretrain_model.model
        self.centroids = pretrain_model.centroids
        self.pretrain_best_eval_score = pretrain_model.best_eval_score

        self.device = model.device
        
        self.train_dataloader = data.dataloader.train_labeled_loader
        self.eval_dataloader = data.dataloader.eval_loader
        self.test_dataloader = data.dataloader.test_loader

        self.loss_fct = loss_map[args.loss_fct]  
        self.best_eval_score = None
        
        if args.train:
            self.delta = None
            self.delta_points = []

        else:
            self.model = restore_model(self.model, args.model_output_dir)
            self.delta = np.load(os.path.join(args.method_output_dir, 'deltas.npy'))
            self.delta = torch.from_numpy(self.delta).to(self.device)
            self.centroids = np.load(os.path.join(args.method_output_dir, 'centroids.npy'))
            self.centroids = torch.from_numpy(self.centroids).to(self.device)

    def set_model_optimizer(self, args, data, model):
        
        self.model = model.set_model(args, 'bert')  
        self.optimizer, self.scheduler = model.set_optimizer(self.model, data.dataloader.num_train_examples, args.train_batch_size, \
                args.num_train_epochs, args.lr, args.warmup_proportion)
        self.device = model.device


    def train(self, args, data):  
        criterion_boundary = BoundaryLoss(num_labels = data.num_labels, feat_dim = args.feat_dim, device = self.device)
        
        self.delta = F.softplus(criterion_boundary.delta)
        self.delta_points.append(self.delta)
        optimizer = torch.optim.Adam(criterion_boundary.parameters(), lr = args.lr_boundary)
        
        if self.centroids is None:
            self.centroids = centroids_cal(self.model, args, data, self.train_dataloader, self.device)
        
        best_eval_score, best_delta, best_centroids = 0, None, None
        wait = 0
        
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            self.model.train()
            # self.model.eval()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            
            for step, batch in enumerate(tqdm(self.train_dataloader, desc="Iteration")):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, _ = batch
                with torch.set_grad_enabled(True):
                    features = self.model(input_ids, segment_ids, input_mask, feature_ext=True)
                    loss, self.delta = criterion_boundary(features, self.centroids, label_ids)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    tr_loss += loss.item()
                    
                    nb_tr_examples += features.shape[0]
                    nb_tr_steps += 1
            print(self.delta)
            self.delta_points.append(self.delta)

            loss = tr_loss / nb_tr_steps
            
            
            if (args.backbone == 'bert_disaware_boost' or args.backbone == 'bert_boost') and (best_eval_score > args.boost_start_score or wait > 0): 
                self.boost(args, data, criterion_boundary, optimizer)
            
            
            y_true, y_pred = self.get_outputs(args, data, mode = 'eval')
            eval_score = round(f1_score(y_true, y_pred, average='macro') * 100, 2)

            eval_results = {
                'train_loss': loss,
                'eval_score': eval_score,
                'best_eval_score':best_eval_score,
            }
            self.logger.info("***** Epoch: %s: Eval results *****", str(epoch + 1))
            for key in sorted(eval_results.keys()):
                self.logger.info("  %s = %s", key, str(eval_results[key]))
            
            if eval_score > best_eval_score:
                wait = 0
                best_delta = self.delta 
                best_eval_score = eval_score
            else:
                if best_eval_score > 0:
                    wait += 1
                    if wait >= args.wait_patient:
                        break

        if best_eval_score > 0:
            self.delta = best_delta
            self.best_eval_score = best_eval_score

        if args.save_model:
            np.save(os.path.join(args.method_output_dir, 'centroids.npy'), self.centroids.detach().cpu().numpy())
            np.save(os.path.join(args.method_output_dir, 'deltas.npy'), self.delta.detach().cpu().numpy())
            np.save(os.path.join(args.method_output_dir, 'all_deltas.npy'), [x.detach().cpu().numpy() for x in self.delta_points])
        
    def boost(self, args, data, criterion_boundary, optimizer):    
        print("---begin boost---")
        ori_examples, labeled_examples, unlabeled_examples = get_examples(args, data.get_attrs(), mode='train')
        method = args.boost_method
        if method.startswith("WP"):
            train_dataloader = self.train_dataloader
            self.model.eval()
            
            total_labels = torch.empty(0,dtype=torch.long).to(self.device)
            total_preds = torch.empty(0,dtype=torch.long).to(self.device)
            
            total_features = torch.empty((0,args.feat_dim)).to(self.device)
            
            
            total_guids = torch.empty(0,dtype=torch.long).to(self.device)
            
            for batch in tqdm(train_dataloader, desc="Iteration"):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, guids = batch
                with torch.set_grad_enabled(False):
                    
                    pooled_output = self.model(input_ids, segment_ids, input_mask, feature_ext=True)
    
                    preds = self.open_classify(data, pooled_output)
                    total_preds = torch.cat((total_preds, preds))
                    total_labels = torch.cat((total_labels, label_ids))
                    total_features = torch.cat((total_features, pooled_output))
    
                    total_guids = torch.cat((total_guids,guids))
                    
            
            y_pred = total_preds.cpu().numpy()
            y_true = total_labels.cpu().numpy()
            
            tagged_guids = []
            for i in range(len(y_pred)):
                if y_true[i] != y_pred[i]: 
                    #print(y_pred[i], y_true[i], total_guids[i])
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
            
        para_examples = paraphrase(para_examples, num_paraphrases = num_paraphrases, cache_file = "%s_train_paraphrases.csv"%args.dataset)
        
        para_dataloader = get_loader(para_examples, args, data.label_list, mode = 'train_labeled', sampler_mode = 'random')
        
        self.model.train()
        
        for step, batch in enumerate(tqdm(para_dataloader, desc="Iteration")):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, _ = batch
            with torch.set_grad_enabled(True):
                features = self.model(input_ids, segment_ids, input_mask, feature_ext=True)
                loss, self.delta = criterion_boundary(features, self.centroids, label_ids)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        
        print("---end boost---")
    def get_outputs(self, args, data, mode = 'eval', get_feats = False, pre_train= False, delta = None):
        
        if mode == 'eval':
            dataloader = self.eval_dataloader
        elif mode == 'test':
            dataloader = self.test_dataloader
        elif mode == 'train':
            dataloader = self.train_dataloader

        self.model.eval()

        total_labels = torch.empty(0,dtype=torch.long).to(self.device)
        total_preds = torch.empty(0,dtype=torch.long).to(self.device)
        
        total_features = torch.empty((0,args.feat_dim)).to(self.device)
        total_logits = torch.empty((0, data.num_labels)).to(self.device)
        
        for batch in tqdm(dataloader, desc="Iteration"):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, _ = batch
            with torch.set_grad_enabled(False):
                
                pooled_output = self.model(input_ids, segment_ids, input_mask, feature_ext=True)

                preds = self.open_classify(data, pooled_output)
                total_preds = torch.cat((total_preds, preds))
                total_labels = torch.cat((total_labels, label_ids))
                total_features = torch.cat((total_features, pooled_output))


        if get_feats:  
            feats = total_features.cpu().numpy()
            return total_features, total_labels
        else:
            y_pred = total_preds.cpu().numpy()
            y_true = total_labels.cpu().numpy()
            return y_true, y_pred

    def open_classify(self, data, features):
        logits = euclidean_metric(features, self.centroids)
        probs, preds = F.softmax(logits.detach(), dim = 1).max(dim = 1)
        euc_dis = torch.norm(features - self.centroids[preds], 2, 1).view(-1)
        preds[euc_dis >= self.delta[preds]] = data.unseen_label_id
        
        return preds
    
    def test(self, args, data, show=True):
        y_true, y_pred = self.get_outputs(args, data, mode = 'test')
        
        cm = confusion_matrix(y_true, y_pred)
        test_results = F_measure(cm)

        acc = round(accuracy_score(y_true, y_pred) * 100, 2)
        test_results['Acc'] = acc
        
        if show:
            self.logger.info("***** Test: Confusion Matrix *****")
            self.logger.info("%s", str(cm))
            self.logger.info("***** Test results *****")
            
            for key in sorted(test_results.keys()):
                self.logger.info("  %s = %s", key, str(test_results[key]))

        test_results['y_true'] = y_true
        test_results['y_pred'] = y_pred
        if args.method == 'DA-ADB:':
            test_results['scale'] = args.scale

        return test_results

    def load_pretrained_model(self, pretrained_model):

        pretrained_dict = pretrained_model.state_dict()
        self.model.load_state_dict(pretrained_dict, strict=False)
