import torch
import matplotlib.pyplot as plt
import numpy as np
from shape_model_hub import ShapeModelHub
from shape_utitls import serialize, deserialize, vec2point
from shape_regularization import TPSM, argument_vec_pts
from shape_simply import PolyReg

# ================== 1. 原始经纬度坐标及序列化 ==================
org_points = [[37, 0], [41, 1], [43, 5], [49, 9], [52, 7], [57, 9], [62, 8], [65, 12], [78, 24], [80, 28], [95, 39], [92, 42], [85, 40], [85, 44], [82, 46], [77, 45], [75, 48], [69, 42], [45, 26], [41, 25], [39, 29], [40, 32], [49, 42], [60, 50], [61, 54], [72, 63], [69, 66], [64, 64], [63, 71], [61, 74], [50, 65], [45, 64], [27, 50], [23, 50], [23, 53], [25, 56], [23, 59], [23, 61], [27, 65], [32, 65], [41, 71], [43, 75], [52, 83], [49, 85], [44, 84], [42, 87], [38, 86], [33, 95], [30, 95], [30, 87], [13, 73], [5, 74], [0, 69], [0, 65], [9, 52], [7, 49], [7, 44], [26, 20], [25, 12], [34, 1], [37, 0]]
# org_points = [
#     [61910.649753771046, 34609.49770304325], [61918.5757181265, 34617.84358927372], [61925.85171910308, 34610.933860269804],
#     [61933.47629185698, 34618.96272989871], [61930.89041539214, 34621.418357340124], [61926.57535191558, 34626.14485880497],
#     [61930.4867288687, 34630.25997110966], [61932.89859410308, 34632.80556437137], [61935.584751329625, 34630.254661051054],
#     [61944.4877054312, 34621.79988810184], [61908.227329454625, 34583.6177591956], [61896.461948595264, 34594.79091593387],
#     [61904.82809849761, 34603.60060831669], [61912.15915318508, 34596.638450113554], [61917.696140489796, 34602.46889444949],
#     [61915.26999302886, 34605.11000773075], [61910.649753771046, 34609.49770304325]
# ]
serialized = serialize(org_points, map_scale=(95, 95))
norm_points = serialized['points']
indexs = serialized['indexs']

# ================== 2. 反归一化（还原） ==================
restored_points = deserialize(norm_points,serialized['params'])

# ================== 3. 规则化与化简 ==================
polyreg = PolyReg()
# PolyReg 规则化：输入为 indexs 的空格分隔字符串，输出为 indexs 字符串
try:
    reg_indexs_str = polyreg.generate(' '.join(str(i) for i in indexs))
    reg_indexs = [int(i) for i in reg_indexs_str.strip().split()]
    reg_points = vec2point(reg_indexs)
    restored_reg_points = deserialize(reg_points,serialized['params'])
except Exception as e:
    print("PolyReg 规则化失败，原因：", e)
    reg_points = None

tpsm = TPSM()
try:
    # TPSM generate 规则化，输入 indexs，输出 indexs 字符串
    tpsm_indexs = serialize(norm_points)['indexs']
    tpsm_indexs_str = tpsm.generate(tpsm_indexs)
    tpsm_indexs = [int(i) for i in tpsm_indexs_str.strip().split()]
    tpsm_points = vec2point(tpsm_indexs)
except Exception as e:
    print("TPSM 规则化失败，原因：", e)
    tpsm_points = None


# ================== 4. 可视化 ==================
def plot_shape(points, title, subplot_idx):
    # 兼容 None、空 list、空 array
    if points is None or (isinstance(points, (list, np.ndarray)) and len(points) < 2):
        plt.subplot(2, 3, subplot_idx)
        plt.title(title + " (无结果)")
        return
    x, y = zip(*points)
    plt.subplot(2, 3, subplot_idx)
    plt.plot(x, y, marker='o')
    plt.title(title)
    plt.axis('equal')

plt.figure(figsize=(15, 8))
plot_shape(org_points, "Original", 1)
plot_shape(norm_points, "Serialized", 2)
plot_shape(restored_points, "Deserialize", 3)
plot_shape(reg_points, "PolyReg", 4)
plot_shape(tpsm_points, "TPSM", 5)
plt.tight_layout()
plt.show()

# ================== 5. 特征编码 ==================
# 使用 serialize 得到的 indexs 作为 token_ids，segment_ids 全 0
token_ids = torch.tensor([serialized['indexs']], dtype=torch.long)  # shape [1, seq_len]
segment_ids = torch.zeros_like(token_ids)  # 单 segment

hub = ShapeModelHub()

# 1. PSRT 编码
try:
    hub.init_psrt('weights/PSRT/config.json', 'weights/PSRT/pytorch_model.bin')
    psrt_feature = hub.psrt_encode(token_ids, segment_ids)
    print("PSRT 编码特征 shape:", psrt_feature.shape)
except Exception as e:
    print("PSRT 编码示例失败，原因：", e)

# 2. ShapeClassifier 编码
try:
    hub.init_shape_classifier('weights/ShapeClassifier/config.json', 'weights/ShapeClassifier/pytorch_model.bin')
    classifier_feature = hub.shape_classifier_encode(token_ids)
    print("ShapeClassifier 编码特征 shape:", classifier_feature.shape)
except Exception as e:
    print("ShapeClassifier 编码示例失败，原因：", e)

# 3. ShapeClassifierBCE 编码
try:
    hub.init_shape_classifier_bce('weights/ShapeClassifierBCE/config.json', 'weights/ShapeClassifierBCE/pytorch_model.bin')
    # ShapeClassifierBCE 支持 batch 输入
    token_ids_list = [token_ids, token_ids]
    bce_feature = hub.shape_classifier_bce_encode(token_ids_list)
    print("ShapeClassifierBCE 编码特征 shape:", bce_feature.shape)
except Exception as e:
    print("ShapeClassifierBCE 编码示例失败，原因：", e)

# ================== 6. Shape Type Prediction (New) ==================
try:
    # 确保已初始化 ShapeClassifier
    if hub.shape_classifier is None:
        hub.init_shape_classifier('weights/ShapeClassifier/config.json', 'weights/ShapeClassifier/pytorch_model.bin')
    pred_type = hub.predict_shape_type_from_latlon(reg_points, ref_json_path='./datasets/split_5010.json')
    print(f"预测的建筑物类型: {pred_type}")
except Exception as e:
    print("Shape type prediction failed:", e)