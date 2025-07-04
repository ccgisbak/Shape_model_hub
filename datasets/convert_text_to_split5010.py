import json
import sys
import os
import numpy as np
def vec2point(vec):
    if vec == None:
        return None
    p = []
    for v in vec:
        p.append([v%96,v//96])
    return np.array(p)

def convert(input_path, output_path):
    """
    将包含'text'字段的样本文件转换为split_5010风格的jsonlines文件
    """
    with open(input_path, 'r', encoding='utf-8') as fin:
        # 支持json数组或jsonlines格式
        try:
            data = json.load(fin)
            if isinstance(data, dict):
                data = data['data']
            print(data[:3])
        except json.JSONDecodeError:
            fin.seek(0)
            data = [json.loads(line) for line in fin if line.strip()]

    with open(output_path, 'w', encoding='utf-8') as fout:
        for item in data:
            text = item.get('text', '')
            if not text:
                continue
            indexs = [int(x) for x in text.strip().split()]
            points = vec2point(indexs)
            # 转为list以便json序列化
            points = points.tolist() if hasattr(points, 'tolist') else points
            out_obj = {
                "indexs": indexs,
                "points": points,
                "type_name": "unknown"
            }
            fout.write(json.dumps(out_obj, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    convert("txt.json","sample.json")