import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from shape_model_hub import ShapeModelHub
from shape_utitls import serialize, vec2point, read_json_points
from PIL import Image, ImageDraw
from bert4torch.tokenizers import Tokenizer
from bert4torch.snippets import sequence_padding

# 递归查找权重和配置文件

def find_file_by_keyword(root_dir, keyword, ext):
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if keyword.lower() in fname.lower() and fname.lower().endswith(ext):
                return os.path.join(dirpath, fname)
    return None

def get_weight_path(model_name, ext='.bin'):
    return find_file_by_keyword('weights', model_name, ext)

def get_config_path(model_name):
    return find_file_by_keyword('weights', model_name, '.json')

def get_test_samples(num=5):
    # 读取部分测试样例
    datafile = 'datasets/sample.json'
    vecs = read_json_points(datafile)
    # 只取前num个
    return [v for v in vecs[:num]]

def visualize_points(points, color=(120,120,200), save_path=None, title=None):
    image = Image.new('RGB', (110, 110), color='white')
    draw = ImageDraw.Draw(image)
    if points is not None and len(points) > 2:
        draw.polygon(tuple(map(tuple, points)), fill=color)
        for p in points:
            draw.point(tuple(map(tuple, [p])), fill=(20,20,120))
    if title:
        plt.title(title)
    if save_path:
        image.save(save_path)
    return image

def show_side_by_side(imgs, titles=None, save_path=None):
    save_path = 'test_results/' + save_path
    n = len(imgs)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    if n == 1:
        axes = [axes]
    for i, img in enumerate(imgs):
        axes[i].imshow(img)
        axes[i].axis('off')
        if titles:
            axes[i].set_title(titles[i])
    if save_path:
        plt.savefig(save_path)
    plt.show()

# 直接指定权重和配置路径
PSRT_CONFIG_PATH = 'weights/psrt/config.json'
PSRT_CKPT_PATH = 'weights/psrt/pytorch_model.bin'
SHAPE_CONFIG_PATH = 'weights/ShapeClassifier/config.json'
SHAPE_CKPT_PATH = 'weights/ShapeClassifier/pytorch_model.bin'
SHAPE_BCE_CONFIG_PATH = 'weights/ShapeClassifierBCE/config.json'
SHAPE_BCE_CKPT_PATH = 'weights/ShapeClassifierBCE/pytorch_model.bin'

def main():
    hub = ShapeModelHub()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # 初始化tokenizer
    vocab_path = './weights/TPSM/vocab.txt'
    tokenizer = Tokenizer(vocab_path, do_lower_case=True)
    maxlen = 64
    
    # 1. PolyReg 测试
    print('=== PolyReg 测试 ===')
    samples = get_test_samples(3)
    for i, pts in enumerate(samples):
        vec = serialize(pts)['indexs']
        vec_str = ' '.join([str(x) for x in vec])
        result_str = hub.polyreg_generate(vec_str)
        result_vec = [int(x) for x in result_str.split() if x.isdigit()]
        result_pts = vec2point(result_vec)
        img1 = visualize_points(pts, color=(120,120,200))
        img2 = visualize_points(result_pts, color=(255,120,120))
        show_side_by_side([img1, img2], ["Original", "regularization"], save_path=f'polyreg_test_{i}.png')

    # 2. TPSM 测试
    print('=== TPSM 测试 ===')
    for i, pts in enumerate(samples):
        vec = serialize(pts)['indexs']
        vec_str = ' '.join([str(x) for x in vec])
        result_str = hub.tpsm_generate(vec_str)
        result_vec = [int(x) for x in result_str.split() if x.isdigit()]
        result_pts = vec2point(result_vec)
        img1 = visualize_points(pts, color=(120,120,200))
        img2 = visualize_points(result_pts, color=(255,120,120))
        show_side_by_side([img1, img2], ["Original", "Simply"], save_path=f'tpsm_test_{i}.png')

    # 3. PSRT_Model 测试
    print('=== PSRT_Model 测试 ===')
    config_path = PSRT_CONFIG_PATH
    checkpoint_path = PSRT_CKPT_PATH
    if os.path.exists(config_path) and os.path.exists(checkpoint_path):
        hub.init_psrt(config_path, checkpoint_path)
        # 构造真实的token_ids和segment_ids
        test_vec = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        token_ids, segment_ids = tokenizer.encode(' '.join([str(i) for i in test_vec]), maxlen=maxlen)
        token_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        segment_ids = torch.tensor([segment_ids], dtype=torch.long, device=device)
        try:
            features = hub.psrt_encode(token_ids, segment_ids)
            print('PSRT_Model编码特征:', features.shape)
        except Exception as e:
            print('PSRT_Model推理失败:', e)
    else:
        print('未找到PSRT模型权重或配置')

    # 4. ShapeClassifier 测试
    print('=== ShapeClassifier 测试 ===')
    config_path = SHAPE_CONFIG_PATH
    checkpoint_path = SHAPE_CKPT_PATH
    if os.path.exists(config_path):
        hub.init_shape_classifier(config_path,checkpoint_path)
        # 构造真实的token_ids
        test_vec = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        token_ids, _ = tokenizer.encode(' '.join([str(i) for i in test_vec]), maxlen=maxlen)
        token_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        try:
            features = hub.shape_classifier_encode(token_ids)
            print('ShapeClassifier编码特征:', features.shape)
        except Exception as e:
            print('ShapeClassifier推理失败:', e)
    else:
        print('未找到ShapeClassifier配置')

    # 5. ShapeClassifierBCE 测试
    print('=== ShapeClassifierBCE 测试 ===')
    config_path = SHAPE_BCE_CONFIG_PATH
    checkpoint_path = SHAPE_BCE_CKPT_PATH
    if os.path.exists(config_path):
        hub.init_shape_classifier_bce(config_path,checkpoint_path)
        # 构造真实的token_ids列表
        test_vec1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        test_vec2 = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        token_ids1, _ = tokenizer.encode(' '.join([str(i) for i in test_vec1]), maxlen=maxlen)
        token_ids2, _ = tokenizer.encode(' '.join([str(i) for i in test_vec2]), maxlen=maxlen)
        token_ids1 = torch.tensor([token_ids1], dtype=torch.long, device=device)
        token_ids2 = torch.tensor([token_ids2], dtype=torch.long, device=device)
        try:
            features = hub.shape_classifier_bce_encode([token_ids1, token_ids2])
            print('ShapeClassifierBCE编码特征:', features.shape)
        except Exception as e:
            print('ShapeClassifierBCE推理失败:', e)
    else:
        print('未找到ShapeClassifierBCE配置')

    # 6. Shape Type Prediction (New)
    print('=== ShapeClassifier 形状类型预测功能测试 ===')
    try:
        if hub.shape_classifier is None:
            hub.init_shape_classifier(SHAPE_CONFIG_PATH, SHAPE_CKPT_PATH)
        # 取一个真实样本
        test_samples = get_test_samples(1)
        if test_samples:
            org_points = test_samples[0]
            pred_type = hub.predict_shape_type_from_latlon(org_points, ref_json_path='./datasets/split_5010.json')
            print(f"预测的建筑物类型: {pred_type}")
        else:
            print("未能获取测试样本")
    except Exception as e:
        print("Shape type prediction test failed:", e)

if __name__ == '__main__':
    main() 