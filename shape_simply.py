from bert4torch.models import build_transformer_model
from bert4torch.tokenizers import Tokenizer, load_vocab
from bert4torch.snippets import sequence_padding, text_segmentate
from bert4torch.snippets import AutoRegressiveDecoder, Callback, ListDataset
import torch
# from torchinfo import summary
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from shape_utitls import *
import numpy as np

maxlen = 128
dict_path = 'd:/DeepLearning/My_Idea/Vec_AE/rebuild_all_in_one/model_saved/huggingface/bert_base/seq2seq/vocab.txt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

token_dict, keep_tokens = load_vocab(
    dict_path=dict_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)
tokenizer = Tokenizer(token_dict, do_lower_case=True)

class PolyReg(AutoRegressiveDecoder):
    """Polygon Regularization Network (PolyReg)
    基于自回归序列生成的建筑轮廓规则化网络。采用Mask-Attention机制，Transformer直接输出结构化规则坐标序列。
    """
    def __init__(self,start_id=None, end_id=tokenizer._token_end_id, maxlen=64, device=device):
        super().__init__(start_id=start_id, end_id=end_id, maxlen=maxlen, device=device)
        self.model = None
        self.load_model()


    def load_model(self,config_path=None,weight_path=None):
        if not self.model==None:
            del self.model
        if config_path==None:
            config_path = config_path = 'd:/DeepLearning/My_Idea/Vec_AE/rebuild_all_in_one/model_saved/huggingface/bert_base/seq2seq/config.json'
        self.model  = build_transformer_model(
                                        config_path,
                                        checkpoint_path=None,
                                        with_mlm='linear',
                                        application='unilm',
                                        keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
                                    ).to(device)
        if weight_path==None:
            weight_path = './weights/PolyReg/' + 'epoch_143_0.25_model.pt'
        self.model.load_weights(weight_path)



    @AutoRegressiveDecoder.wraps(default_rtype='logits')
    def predict(self, inputs, output_ids, states):
        token_ids, segment_ids = inputs
        token_ids = torch.cat([token_ids, output_ids], 1)
        segment_ids = torch.cat([segment_ids, torch.ones_like(output_ids, device=device)], 1)
        _, y_pred = self.model.predict([token_ids, segment_ids])
        # print(y_pred.shape)
        return y_pred[:, -1, :]

    def generate(self, text, topk=3, topp=0.95):
        max_c_len = maxlen - self.maxlen
        token_ids, segment_ids = tokenizer.encode(text, maxlen=max_c_len)
        output_ids = self.beam_search([token_ids, segment_ids], topk=topk)  # 基于beam search
        # print(output_ids,tokenizer.decode(output_ids.cpu().numpy()))
        return tokenizer.decode(output_ids.cpu().numpy())


def argument_vec_pts(vec_p,flag=True):
    argument_vec = argument_by_all_index(vec_p,list(range(len(vec_p))[1:])) if flag else vec_p
    argument_vec = argument_by_PIL(argument_vec)
    return argument_vec


class load_data:
    def __init__(self,sample_num=8) -> None:
        datafile_name = "database/regular_urban.json"
        try:
            self.vecs = read_json(datafile_name)
        except:
            self.vecs = read_txt(datafile_name)
        self.load_source_vecpts(sample_num)



    def load_source_vecpts(self,sample_num=8):
        vecs = self.vecs
        vecs = [vec for vec in vecs if len(vec) > 8 ]
        vec_array = np.random.choice(vecs,sample_num)
        vec_p = [vec2point(vec) for vec in vec_array]
        argument_vec = [argument_vec_pts(v,1) for v in vec_p]
        self.source_vecpts = argument_vec

# vec_data = load_data()

# regular_urban = PolyReg(start_id=None, end_id=tokenizer._token_end_id, maxlen=64, device=device)
# show_vec_data = []
# for vec in vec_data.source_vecpts:
#     vec = serialize(vec)
#     vec = vec['indexs']        
#     test_vec = ' '.join([str(i) for i in vec])
#     show_vec_data.append([int(i) for i in test_vec.split(' ')])
#     result_vec = regular_urban.generate(test_vec)
#     sv = [int(i) for i in result_vec.split(' ')]
#     show_vec_data.append(sv)
# show_vec_list(show_vec_data)
# a = input()
