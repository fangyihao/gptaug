from .bert import BERT, BERT_Norm, BERT_K_1_way, BERT_SEG, BERT_Disaware, BERT_DOC, BERT_MDF, BERT_MDF_Pretrain

backbones_map = {
                    'bert': BERT, 
                    'bert_norm': BERT_Norm,
                    'bert_K+1-way': BERT_K_1_way,
                    'bert_seg': BERT_SEG,
                    'bert_disaware': BERT_Disaware,
                    'bert_doc': BERT_DOC,
                    'bert_mdf': BERT_MDF,
                    'bert_mdf_pretrain': BERT_MDF_Pretrain,
                    'bert_disaware_boost': BERT_Disaware,
                    'bert_boost': BERT
                }
