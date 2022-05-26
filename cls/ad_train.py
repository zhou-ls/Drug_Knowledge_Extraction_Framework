# -*- coding: utf-8 -*-
# @Time : 2020/11/29 14:44
# @Author : lensen
# @File : ad_train.py
# 通过对抗训练增强模型的泛化性能
# 比CLUE榜单公开的同数据集上的BERT base的成绩高2%
# 适用于Keras 2.3.1

import json
import logging
import numpy as np
from bert4keras.backend import keras, search_layer, K
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from keras import Model
from keras.layers import Lambda, Dense, Dropout, Concatenate
from tqdm import tqdm
from utils import train_file_path, test_file_path, log_path, MAX_SEQ_LEN, batch_size, num_classes, epoch, dict_path, \
    config_path, checkpoint_path, start_at


# num_classes = 2
# maxlen = 128
# batch_size = 32

# # BERT base
# config_path = 'bert/chinese_L-12_H-768_A-12/bert_config.json'
# checkpoint_path = 'bert/chinese_L-12_H-768_A-12/bert_model.ckpt'
# dict_path = 'bert/chinese_L-12_H-768_A-12/vocab.txt'


def load_data(filename):
    """加载数据
    单条格式：(标签, 文本)
    """
    D = []
    with open(filename, 'r', encoding='utf-8') as f:
        content = [_.strip() for _ in f.readlines()]
    for line in content:
        parts = line.split()
        label, text = parts[0], ''.join(parts[1:])
        D.append((text, int(label)))
    return D


# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=MAX_SEQ_LEN)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


# 加载数据集
train_data = load_data(train_file_path)
valid_data = load_data(test_file_path)
# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)

# # 加载预训练模型
# bert = build_transformer_model(
#     config_path=config_path,
#     checkpoint_path=checkpoint_path,
#     return_keras_model=False,
# )
#
# output = Lambda(lambda x: x[:, 0])(bert.model.output)
# output = Dense(
#     units=num_classes,
#     activation='softmax',
#     kernel_initializer=bert.initializer
# )(output)
#
# model = keras.model.Model(bert.model.input, output)
# model.summary()
#
# model.compile(
#     loss='sparse_categorical_crossentropy',
#     optimizer=Adam(2e-5),
#     metrics=['accuracy'],
# )


def build_model(mode='bert', lastfour=False, LR=1e-5, DR=0.2):
    global tokenizer
    tokenizer = Tokenizer(dict_path, do_lower_case=True)

    bert = build_transformer_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        with_pool=True,
        model=mode,
        return_keras_model=False,
    )
    if lastfour:
        model = Model(
            inputs=bert.model.input,
            outputs=[
                bert.model.layers[-3].get_output_at(0),
                bert.model.layers[-11].get_output_at(0),
                bert.model.layers[-19].get_output_at(0),
                bert.model.layers[-27].get_output_at(0),
            ]
        )
        output = model.outputs
        output1 = Lambda(lambda x: x[:, 0], name='Pooler1')(output[0])
        output2 = Lambda(lambda x: x[:, 0], name='Pooler2')(output[1])
        output3 = Lambda(lambda x: x[:, 0], name='Pooler3')(output[2])
        output4 = Lambda(lambda x: x[:, 0], name='Pooler4')(output[3])

        output = Concatenate(axis=1)([output1, output2, output3, output4])

    else:
        output = bert.model.output

    output = Dropout(rate=DR)(output)
    output = Dense(units=num_classes,
                   activation='softmax',
                   kernel_initializer=bert.initializer)(output)

    model = Model(bert.model.input, output)
    print(model.summary())
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(LR),
        metrics=['accuracy'],
    )
    return model


def adversarial_training(model, embedding_name, epsilon=1.0):
    """给模型添加对抗训练
    其中model是需要添加对抗训练的keras模型，embedding_name
    则是model里边Embedding层的名字。要在模型compile之后使用。
    """
    if model.train_function is None:  # 如果还没有训练函数
        model._make_train_function()  # 手动make
    old_train_function = model.train_function  # 备份旧的训练函数

    # 查找Embedding层
    for output in model.outputs:
        embedding_layer = search_layer(output, embedding_name)
        if embedding_layer is not None:
            break
    if embedding_layer is None:
        raise Exception('Embedding layer not found')

    # 求Embedding梯度
    embeddings = embedding_layer.embeddings  # Embedding矩阵
    gradients = K.gradients(model.total_loss, [embeddings])  # Embedding梯度
    gradients = K.zeros_like(embeddings) + gradients[0]  # 转为dense tensor

    # 封装为函数
    inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)  # 所有输入层
    embedding_gradients = K.function(
        inputs=inputs,
        outputs=[gradients],
        name='embedding_gradients',
    )  # 封装为函数

    def train_function(inputs):  # 重新定义训练函数
        grads = embedding_gradients(inputs)[0]  # Embedding梯度
        delta = epsilon * grads / (np.sqrt((grads**2).sum()) + 1e-8)  # 计算扰动
        K.set_value(embeddings, K.eval(embeddings) + delta)  # 注入扰动
        outputs = old_train_function(inputs)  # 梯度下降
        K.set_value(embeddings, K.eval(embeddings) - delta)  # 删除扰动
        return outputs

    model.train_function = train_function  # 覆盖原训练函数


model = build_model(mode='bert', lastfour=True, LR=1e-5, DR=0.2)
# 写好函数后，启用对抗训练只需要一行代码
if start_at:
    adversarial_training(model, 'Embedding-Token', 0.5)


def evaluate(data):
    total, right = 0., 0.
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]
        total += len(y_true)
        right += (y_true == y_pred).sum()
    return right / total


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        super().__init__()
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(valid_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            model.save_weights('./model/ad_train/best_model.weights')
            # model.save_weights('./model/ad_train/epoch%d-best_val_acc%.5f.weights' % (epoch, self.best_val_acc))
        print(u'val_acc: %.5f, best_val_acc: %.5f\n' % (val_acc, self.best_val_acc))
        logging.basicConfig(level=logging.DEBUG,  # 控制台打印的日志级别
                            filename=log_path,
                            filemode='a',  # 模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志,a是追加模式，默认如果不写的话，就是追加模式
                            format='%(asctime)s - %(levelname)s: %(message)s\n'  # 日志格式
                            )
        logging.info(u'val_acc: %.5f, best_val_acc: %.5f\n' % (val_acc, self.best_val_acc))


def train():
    evaluator = Evaluator()

    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epoch,
        callbacks=[evaluator]
    )


def predict_to_file(in_file, out_file):
    """
    输出预测结果到文件
    """
    fw = open(out_file, 'w')
    with open(in_file) as fr:
        for l in tqdm(fr):
            l = json.loads(l)
            text = l['sentence']
            token_ids, segment_ids = tokenizer.encode(text, maxlen=MAX_SEQ_LEN)
            label = model.predict([[token_ids], [segment_ids]])[0].argmax()
            l = json.dumps({'id': str(l['id']), 'label': str(label)})
            fw.write(l + '\n')
    fw.close()


if __name__ == '__main__':
    train()
else:
    model.load_weights('./model/ad_train/best_model.weights')
    # predict_to_file(test_file_path, 'iflytek_predict.json')
