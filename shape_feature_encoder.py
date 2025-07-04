from bert4torch.tokenizers import Tokenizer
from bert4torch.models import build_transformer_model, BaseModel
from bert4torch.snippets import sequence_padding, Callback, ListDataset
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
from scipy.stats import pearsonr, spearmanr
import copy,json
import random



class ShapeClassifierBCE(BaseModel):
    def __init__(self,config_path, pool_method='mean', scale=20.0):
        super().__init__()
        self.model1, self.config = build_transformer_model(config_path=config_path, with_pool=True, return_model_config=True, segment_vocab_size=0)
        self.model2 = copy.deepcopy(self.model1)
        self.pool_method = pool_method
        self.scale = scale

    def forward(self, token_ids_list):
        token_ids = token_ids_list[0]
        hidden_state1, pool_cls1 = self.model1([token_ids])
        embeddings_a = self.get_pool_emb(hidden_state1, pool_cls1, attention_mask=token_ids.gt(0).long())

        token_ids = token_ids_list[1]
        hidden_state2, pool_cls2 = self.model2([token_ids])
        embeddings_b = self.get_pool_emb(hidden_state2, pool_cls2, attention_mask=token_ids.gt(0).long())

        scores = self.cos_sim(embeddings_a, embeddings_b) * self.scale  # [btz, btz]
        return scores

    def encode(self, token_ids,no_use=None):
        self.eval()
        with torch.no_grad():
            hidden_state, pool_cls = self.model1([token_ids])
            output = self.get_pool_emb(hidden_state, pool_cls, attention_mask=token_ids.gt(0).long())
        return output
    
    def get_pool_emb(self, hidden_state, pool_cls, attention_mask):
        if self.pool_method == 'cls':
            return pool_cls
        elif self.pool_method == 'mean':
            hidden_state = torch.sum(hidden_state * attention_mask[:, :, None], dim=1)
            attention_mask = torch.sum(attention_mask, dim=1)[:, None]
            return hidden_state / attention_mask
        elif self.pool_method == 'max':
            seq_state = hidden_state * attention_mask[:, :, None]
            return torch.max(seq_state, dim=1)
        else:
            raise ValueError('pool_method illegal')
    
    # def load_weights(self, load_path, strict=True, prefix=None):
    #     self.model1.load_weights_from_pytorch_checkpoint(load_path)
    #     self.model2.load_weights_from_pytorch_checkpoint(load_path)
        

    @staticmethod
    def cos_sim(a, b):
        a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
        b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
        return torch.mm(a_norm, b_norm.transpose(0, 1))



class ShapeClassifier(BaseModel):
    def __init__(self, config_path,pool_method='mean'):
        super().__init__()
        self.bert, self.config = build_transformer_model(config_path=config_path, with_pool=True, return_model_config=True, segment_vocab_size=0)
        self.pool_method = pool_method

    def forward(self, token1_ids, token2_ids):
        hidden_state1, pool_cls1 = self.bert([token1_ids])
        pool_emb1 = self.get_pool_emb(hidden_state1, pool_cls1, attention_mask=token1_ids.gt(0).long())
        
        hidden_state2, pool_cls2 = self.bert([token2_ids])
        pool_emb2 = self.get_pool_emb(hidden_state2, pool_cls2, attention_mask=token2_ids.gt(0).long())

        return 1- torch.cosine_similarity(pool_emb1, pool_emb2)
    
    def get_pool_emb(self, hidden_state, pool_cls, attention_mask):
        if self.pool_method == 'cls':
            return pool_cls
        elif self.pool_method == 'mean':
            hidden_state = torch.sum(hidden_state * attention_mask[:, :, None], dim=1)
            attention_mask = torch.sum(attention_mask, dim=1)[:, None]
            return hidden_state / attention_mask
        elif self.pool_method == 'max':
            seq_state = hidden_state * attention_mask[:, :, None]
            return torch.max(seq_state, dim=1)
        else:
            raise ValueError('pool_method illegal')

    def encode(self, token_ids,no_use=None):
        self.eval()
        with torch.no_grad():
            hidden_state, pool_cls = self.bert([token_ids])
            attention_mask = token_ids.gt(0).long()
            output = self.get_pool_emb(hidden_state, pool_cls, attention_mask)
        return output
    

class PSRT_Model(BaseModel):
    def __init__(self, config_path,checkpoint_path,pool_method='mean'):
        super().__init__()
        self.bert = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, with_mlm=True)
        self.pool_method = pool_method

    def get_pool_emb(self, hidden_state, pool_cls, attention_mask):
        if self.pool_method == 'cls':
            return pool_cls
        elif self.pool_method == 'mean':
            # # hidden_state = torch.sum(hidden_state * attention_mask[:, :, None], dim=1)
            # # attention_mask = torch.sum(attention_mask, dim=1)[:, None]
            # # return hidden_state / attention_mask
            # # return torch.mean(hidden_state,dim=1)
            hidden_state = torch.sum(hidden_state * attention_mask[:, :, None], dim=1)
            attention_mask = torch.sum(attention_mask, dim=1)[:, None]
            return hidden_state / attention_mask            
        elif self.pool_method == 'max':
            seq_state = hidden_state * attention_mask[:, :, None]
            return torch.max(seq_state, dim=1)
        else:
            raise ValueError('pool_method illegal')

    def forward(self, token_ids, segment_ids):
        hidden_state, pool_cls = self.bert([token_ids, segment_ids])
        sen_emb = self.get_pool_emb(hidden_state, pool_cls, attention_mask=token_ids.gt(0).long())
        print(pool_cls.shape,hidden_state.shape)
        return None, hidden_state[:,0,:] #sen_emb#

    def encode(self, token_ids,segment_ids):
        self.eval()
        with torch.no_grad():
            # segment_ids = torch.zeros_like(token_ids)
            hidden_state,pool_cls = self.bert([token_ids,segment_ids])
            # print(pool_cls.shape)
            # hidden_state = pool_cls
            attention_mask = token_ids.gt(0).long()
            output = self.get_pool_emb(hidden_state, pool_cls, attention_mask)
            print(pool_cls.shape,hidden_state.shape,output.shape)
        return output        