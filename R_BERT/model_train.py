# -*- coding: utf-8 -*-

import os
import json
# import tensorflow._api.v2.compat.v1 as tf
# tf.disable_v2_behavior()
import codecs
import pandas as pd
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from keras_bert import Tokenizer
from keras_bert import AdamWarmup, calc_train_steps
from tensorflow.keras.callbacks import ModelCheckpoint


from model import RBERT
from config import *

token_dict = {}
with codecs.open(BERT_VOCAT_PATH, 'r', 'utf-8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)


class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            else:
                R.append('[UNK]')   # 剩余的字符是[UNK]
        return R


tokenizer = OurTokenizer(token_dict)


def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


class DataGenerator:

    def __init__(self, data, batch_size=BATCH_SIZE):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            np.random.shuffle(idxs)
            X1, X2, MASK1, MASK2, Y = [], [], [], [], []
            for i in idxs:
                d = self.data[i]
                # text = d[0].replace("$", "").replace("*", "")
                # text = text.replace("</e1>", "$").replace("</e2>", "*")
                x1, x2 = tokenizer.encode(first=d[0], max_len=MAX_LEN)
                x1[x1.index(102)] = 0   # 去掉[SEP]
                X1.append(x1)
                X2.append(x2)
                # print(d)
                # 寻找$,#的下标
                special_index_1 = [i for i in range(len(x1)) if x1[i] == 109]   # index of $
                # special_index_2 = [i for i in range(len(x1)) if x1[i] == 115]   # index of *
                # assert len(special_index_1) == 2, "$在文本中的数量不为2，请检查: {}".format(text)#做了修改
                # assert len(special_index_2) == 2, "#在文本中的数量不为2，请检查: {}".format(text)

                mask1 = [0] * MAX_LEN
                start1 = 0
                end1 = special_index_1[0]
                for i in range(start1, end1):
                    mask1[i] = 1 / (end1-start1-1)
                mask2 = [0] * MAX_LEN
                start2=special_index_1[0]
                end2 = special_index_1[1]#做了修改
                for i in range(start2+1, end2):
                    mask2[i] = 1 / (end2-start2-1)
                MASK1.append(mask1)
                MASK2.append(mask2)

                y = d[1]
                Y.append(y)
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    MASK1 = np.array(MASK1)
                    MASK2 = np.array(MASK2)
                    Y = seq_padding(Y)
                    yield [X1, X2, MASK1, MASK2], Y
                    [X1, X2, MASK1, MASK2, Y] = [], [], [], [], []


# 构建模型
def create_cls_model(num_labels):
    model = RBERT(BERT_CONFIG_PATH, BERT_CHECKPOINT_PATH, MAX_LEN, num_labels).create_model()
    return model

import random
if __name__ == '__main__':

    # 数据处理, 读取训练集和测试集
    print("begin data processing...")
    train_df=pd.read_excel(TRAIN_FILE_PATH, header=None)
    test_df = pd.read_excel(TEST_FILE_PATH, header=None)

    # train_df = pd.read_csv(TRAIN_FILE_PATH, sep="\t", header=None)
    # test_df = pd.read_csv(TEST_FILE_PATH, sep="\t", header=None)
    labels = train_df[0].unique()
    labels = [str(x) for x in labels]
    with open(os.path.join(DATA_DIR, "label.json"), "w", encoding="utf-8") as f:
        f.write(json.dumps(dict(zip(range(len(labels)), labels)), ensure_ascii=False, indent=2))

    train_data = []
    test_data = []
    for i in range(train_df.shape[0]):
        label, content = train_df.iloc[i, :]
        label_id = [0] * len(labels)
        for j, _ in enumerate(labels):
            if _ == label:
                label_id[j] = 1
        train_data.append((content, label_id))

    for i in range(test_df.shape[0]):
        label, content = test_df.iloc[i, :]
        label_id = [0] * len(labels)
        for j, _ in enumerate(labels):
            if _ == label:
                label_id[j] = 1
        test_data.append((content, label_id))

    print("finish data processing!")

    # 模型训练
    model = create_cls_model(len(labels))
    train_D = DataGenerator(train_data)
    test_D = DataGenerator(test_data)
    print("begin model training...")
    # 保存最新的val_acc最好的模型文件
    filepath = "models/%s-{epoch:02d}-{val_acc:.4f}.h5" % DATA_DIR.split("/")[-1]
    early_stopping = EarlyStopping(monitor='val_acc', patience=5, mode='max')
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    # add warmup
    total_steps, warmup_steps = calc_train_steps(
        num_example=len(train_data),
        batch_size=BATCH_SIZE,
        epochs=EPOCH,
        warmup_proportion=0.1,
    )
    optimizer = AdamWarmup(total_steps, warmup_steps, lr=5e-5, min_lr=1e-7)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    model.fit_generator(
        train_D.__iter__(),
        steps_per_epoch=len(train_D),
        epochs=EPOCH,
        validation_data=test_D.__iter__(),
        validation_steps=len(test_D),
        callbacks=[checkpoint, early_stopping]
    )
    print("finish model training!")

    result = model.evaluate_generator(test_D.__iter__(), steps=len(test_D))
    print("模型评估结果:", result)
