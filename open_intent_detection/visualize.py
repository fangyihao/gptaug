from run import *
from utils.functions import *
from backbones.bert import BERT_Disaware, CosNorm_Classifier, BERT
from transformers import AutoConfig
import pdb

import sys

sys.path.append('.')
args = parse_arguments()
args.method = 'DA-ADB'
args.backbone = 'bert_disaware'
args.config_file_name = 'DA-ADB'
args.dataset = 'banking'
args.known_cls_ratio = 0.75
args.device = "cuda:0"

logger = set_logger(args)

logger.info('Open Intent Detection Begin...')
logger.info('Parameters Initialization...')
param = ParamManager(args)
args = param.args
print(args)
data = DataManager(args, logger_name = args.logger_name)


#print(data.dataloader.test_examples[10].label)
#pdb.set_trace()
test_dataloader = data.dataloader.test_loader


model = BERT_Disaware.from_pretrained('bert-base-uncased', cache_dir = "cache", args = args)
model = restore_model(model, '/home/yfang/workspace/TEXTOIR/output/open_intent_detection/DA-ADB_banking_0.75_1.0_bert_disaware_0/models') # 
model.to('cuda')



def get_outputs(dataloader):

    model.eval()
    total_labels = torch.empty(0, dtype=torch.long).to('cuda')
    total_preds = torch.empty(0, dtype=torch.long).to('cuda')
    
    total_features = torch.empty((0, args.feat_dim)).to('cuda')
    total_logits = torch.empty((0, data.num_labels)).to('cuda')
    
    centroids = centroids_cal(model, args, data, test_dataloader, device='cuda')
    
    for batch in tqdm(dataloader, desc="Iteration"):
    
        batch = tuple(t.to('cuda') for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch
        #input_ids = input_ids.squeeze(0)
        #print(len(label_ids))
        with torch.set_grad_enabled(False):
        
            pooled_output, logits = model(input_ids, segment_ids, input_mask, centroids = centroids, labels = label_ids, mode = 'eval')
            
            total_labels = torch.cat((total_labels, label_ids))
            total_features = torch.cat((total_features, pooled_output))
            total_logits = torch.cat((total_logits, logits))
            
        total_probs = torch.nn.functional.softmax(total_logits.detach(), dim=1)
        total_maxprobs, total_preds = total_probs.max(dim=1)
        
    maxprobs = total_maxprobs.cpu().numpy()
    y_pred = total_preds.cpu().numpy()
    y_true = total_labels.cpu().numpy()
    
    return y_true, y_pred, maxprobs

y_true, y_pred, maxprobs = get_outputs(test_dataloader)

print(list(y_pred))
print(list(y_true))
print(list(maxprobs))


name_list = 0
for i in range(len(y_pred)):
    if y_true[i] != y_pred[i] and maxprobs[i]>0.3:
        print(y_pred[i], y_true[i], i)
        name_list += 1


print(name_list)



# from sklearn.manifold import TSNE
# T_trans = TSNE(n_components=2)
# low_dim_data = T_trans.fit_transform(np.array(emb.cpu()))
# print('Lower dim data has shape',low_dim_data.shape)
#
# # set some styles for for Plotting
# import seaborn as sns
# # Style Plots a bit
# sns.set_style('darkgrid')
# sns.set_palette('muted')
# sns.set_context("notebook", font_scale=1,rc={"lines.linewidth": 2.5})
#
# import matplotlib as plt
# plt.rcParams['figure.figsize'] = (20, 14)
# from matplotlib import pyplot as plt
#
# tsne_df =  pd.DataFrame(low_dim_data, np.array(labels.cpu().numpy()))
# tsne_df.columns = ['x','y']
# print(tsne_df.head(10))
#
# ax = sns.scatterplot(data=tsne_df, x='x', y='y', hue=tsne_df.index)
# ax.set_title('T-SNE BERT Embeddings, PFT-ADB')
# plt.savefig('C:/important/queens/lab/summer project/DA-ADB/TEXTOIR/open_intent_detection/models/ADB/ADB.png')
