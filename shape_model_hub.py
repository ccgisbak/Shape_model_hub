from shape_utitls import read_json_points, vec2point, serialize
from shape_simply import PolyReg, load_data as load_polyreg_data
from shape_regularization import TPSM
from shape_feature_encoder import PSRT_Model, ShapeClassifier, ShapeClassifierBCE
import torch
from bert4torch.tokenizers import Tokenizer

class ShapeModelHub:
    """
    统一入口，管理和调用所有建筑形状相关模型。
    """
    def __init__(self):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        # 初始化各类模型（可根据实际需求传递参数）
        self.polyreg = PolyReg()
        self.polyreg.tokenizer = Tokenizer('weights/PolyReg/vocab.txt')
        self.tpsm = TPSM()
        self.tpsm.tokenizer = Tokenizer('weights/TPSM/vocab.txt')
        self.psrt = None  # 需传入config_path、checkpoint_path
        self.psrt_tokenizer = None
        self.shape_classifier = None  # 需传入config_path
        self.shape_classifier_tokenizer = None
        self.shape_classifier_bce = None  # 需传入config_path
        self.shape_classifier_bce_tokenizer = None

    def init_psrt(self, config_path, checkpoint_path, pool_method='mean'):
        self.psrt = PSRT_Model(config_path, checkpoint_path, pool_method)
        self.psrt.to(self.device)
        self.psrt.tokenizer = Tokenizer('weights/PSRT/vocab.txt')
        self.psrt_tokenizer = self.psrt.tokenizer

    def init_shape_classifier(self, config_path, checkpoint_path, pool_method='mean'):
        self.shape_classifier = ShapeClassifier(config_path, pool_method)
        self.shape_classifier.load_weights(checkpoint_path)
        self.shape_classifier.to(self.device)
        self.shape_classifier.tokenizer = Tokenizer('weights/ShapeClassifier/vocab.txt')
        self.shape_classifier_tokenizer = self.shape_classifier.tokenizer

    def init_shape_classifier_bce(self, config_path, checkpoint_path, pool_method='mean', scale=20.0):
        self.shape_classifier_bce = ShapeClassifierBCE(config_path, pool_method, scale)
        self.shape_classifier_bce.load_weights(checkpoint_path)
        self.shape_classifier_bce.to(self.device)
        self.shape_classifier_bce.tokenizer = Tokenizer('weights/ShapeClassifierBCE/vocab.txt')
        self.shape_classifier_bce_tokenizer = self.shape_classifier_bce.tokenizer

    def polyreg_generate(self, text, topk=3, topp=0.95):
        return self.polyreg.generate(text, topk, topp)

    def tpsm_generate(self, vec, topk=3, topp=0.95):
        return self.tpsm.generate(vec, topk, topp)

    def psrt_encode(self, token_ids, segment_ids):
        if self.psrt is None:
            raise ValueError('PSRT_Model未初始化')
        # 确保输入tensor在正确的设备上
        token_ids = token_ids.to(self.device)
        segment_ids = segment_ids.to(self.device)
        return self.psrt.encode(token_ids, segment_ids)

    def shape_classifier_encode(self, token_ids):
        if self.shape_classifier is None:
            raise ValueError('ShapeClassifier未初始化')
        # 确保输入tensor在正确的设备上
        token_ids = token_ids.to(self.device)
        return self.shape_classifier.encode(token_ids)

    def shape_classifier_bce_encode(self, token_ids_list):
        if self.shape_classifier_bce is None:
            raise ValueError('ShapeClassifierBCE未初始化')
        import torch
        # Ensure all are tensors and on the correct device
        token_ids_list = [torch.tensor(t) if not isinstance(t, torch.Tensor) else t for t in token_ids_list]
        token_ids_list = [t.to(self.device) for t in token_ids_list]
        # Stack or concatenate into a batch tensor of shape [batch, seq_len]
        token_ids_batch = torch.cat(token_ids_list, dim=0)
        return self.shape_classifier_bce.encode(token_ids_batch)

    # 可根据需要添加更多统一入口方法

    def preprocess_coords(self, all_points, map_scale=(95, 95)):
        """
        预处理：对输入经纬度坐标点进行归一化（序列化），返回归一化结果和参数。
        """
        from shape_utitls import serialize
        return serialize(all_points)

    def postprocess_coords(self, serialized_border):
        """
        后处理：根据归一化参数，将处理后的坐标还原为原始坐标。
        """
        from shape_utitls import deserialize
        return deserialize(serialized_border)

    def predict_shape_type_from_latlon(self, input_points, ref_json_path='./datasets/split_5010.json', map_scale=(95, 95)):
        """
        预测输入建筑物轮廓（原始经纬度坐标）的形状类别。
        :param input_points: list of [x, y] 原始经纬度坐标
        :param ref_json_path: 参考形状的json文件路径
        :param map_scale: 归一化尺度
        :return: 预测类别（如 'E', 'F', ...）
        """
        import torch
        import random
        from shape_utitls import read_json_types, serialize,simpfyTrans

        # 1. 加载参考数据，每类随机选一个
        all_data = read_json_types(ref_json_path)
        ref_types = list(all_data.keys())
        ref_samples = {}
        for t in ref_types:
            ref_samples[t] = random.choice(all_data[t])
        # print(ref_samples)

        # 2. 对参考和输入形状做序列化
        ref_vecs = []
        ref_labels = []
        for t, sample in ref_samples.items():
            ser = serialize(sample['org_points'])
            if ser is None:
                continue
            ref_vecs.append(sample['indexs'])
            ref_labels.append(t)

        input_ser = serialize(input_points)
        if input_ser is None:
            raise ValueError("Input points serialization failed.")
        input_vec = simpfyTrans(input_ser['indexs'])

        # 3. 编码并padding
        from bert4torch.snippets import sequence_padding
        all_vecs = ref_vecs + [input_vec]
        X, S = [], []
        # 编码每个vec
        for t in all_vecs:
            x, s = self.shape_classifier.tokenizer.encode(' '.join([str(i) for i in t]))
            X.append(x)
            S.append(s)
        maxlen = max(len(x) for x in X)
        X = torch.tensor(sequence_padding(X, length=maxlen), dtype=torch.long, device=self.device)
        S = torch.tensor(sequence_padding(S, length=maxlen), dtype=torch.long, device=self.device)

        # 4. 编码
        if self.shape_classifier is None:
            raise ValueError('ShapeClassifier未初始化')
        with torch.no_grad():
            # 如果 shape_classifier.encode 支持 segment_ids，传入X和S，否则只传X
            try:
                embeddings = self.shape_classifier.encode(X, S)
            except TypeError:
                embeddings = self.shape_classifier.encode(X)
        ref_embs = embeddings[:-1]
        input_emb = embeddings[-1].unsqueeze(0)  # [1, dim]

        # 5. 计算余弦相似度
        # input_emb_norm = torch.nn.functional.normalize(input_emb, p=2, dim=1)
        # ref_embs_norm = torch.nn.functional.normalize(ref_embs, p=2, dim=1)
        # print(ref_embs.shape,input_emb.shape)
        # print(input_emb_norm.shape,ref_embs_norm.shape)
        sims = torch.matmul(input_emb, ref_embs.t()).squeeze(0)  # [num_types]

        # 6. 取最大相似度的类别
        best_idx = torch.argmax(sims).item()
        return ref_labels[best_idx],sims

# 示例用法
if __name__ == '__main__':
    hub = ShapeModelHub()
    # hub.init_psrt('path/to/config.json', 'path/to/checkpoint.bin')
    # hub.init_shape_classifier('path/to/config.json')
    # hub.init_shape_classifier_bce('path/to/config.json')
    # result = hub.polyreg_generate('1 2 3 4 5')
    # print(result) 