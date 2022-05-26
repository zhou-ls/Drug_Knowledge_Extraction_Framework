import json
import os
import numpy as np
from keras.models import load_model
from att import Attention
from draw_picture import draw_p_r
from utils import model_dir, json_label2id, model_name, evaluate_file

if model_name == 'albert':
    from albert_zh.extract_feature import BertVector
elif model_name == 'bert':
    from bert.extract_feature import BertVector
elif model_name == 'bert_wwm':
    from bert_wwm.extract_feature import BertVector
elif model_name == 'bert_wwm_ext':
    from bert_wwm_ext.extract_feature import BertVector
elif model_name == 'ELECTRA':
    from ELECTRA.extract_feature import BertVector
elif model_name == 'roberta':
    from roberta.extract_feature import BertVector
elif model_name == 'roberta_wwm_large':
    from roberta_wwm_large.extract_feature import BertVector
elif model_name == 'ernie':
    from Ernie.extract_feature import BertVector

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 加载训练效果最好的模型
files = os.listdir(model_dir)
models_path = [os.path.join(model_dir, _) for _ in files]
# 取训练效果最好的模型名
best_model_path = sorted(models_path, key=lambda x: float(x.split('-')[-1].replace('.h5', '')), reverse=True)[0]
model = load_model(best_model_path, custom_objects={"Attention": Attention})
bert_model = BertVector(pooling_strategy="NONE", max_seq_len=80)

with open(json_label2id, 'r', encoding='utf-8') as f:
    rel_dict = json.load(f)

y_true = []
y_score = []


def predict(path):
    total, n = 0, 0
    file = open(path, 'r', encoding='utf-8')
    while True:
        line = file.readline()
        if line == '':
            break
        unit = line.strip().split()
        data = '#'.join([unit[0], unit[1], unit[3]])  # 实体1#实体2#文本信息
        ex = unit[2]  # 关系
        total += 1
        # sen:实体1#实体2#文本信息
        en1, en2, sen = data.split('#')
        text = '$'.join([en1, en2, sen.replace(en1, len(en1) * '#').replace(en2, len(en2) * '#')])
        # 利用BERT提取句子特征
        vec = bert_model.encode([text])["encodes"][0]
        x_train = np.array([vec])

        # 模型预测并输出预测结果
        predicted = model.predict(x_train)
        y = np.argmax(predicted[0])

        id_rel_dict = {v: k for k, v in rel_dict.items()}
        if id_rel_dict[y] != ex:
            y_true.append(0)
            y_score.append(list(predicted[0])[y])
            n += 1
            print(f"原文: {text}\n"
                  f"标注的实体关系: {ex}\n"
                  f"预测错误的实体关系: {id_rel_dict[y]}\n"
                  f"置信度：{list(predicted[0])[y]:.4f}\n")
        else:
            y_true.append(1)
            y_score.append(list(predicted[0])[y])

    print(f"总共有{total}条数据\n"
          f"预测错误{n}条数据\n"
          f"正确率：{1 - n / total}")


if __name__ == '__main__':
    predict(path=evaluate_file)
    draw_p_r(y_true, y_score)
