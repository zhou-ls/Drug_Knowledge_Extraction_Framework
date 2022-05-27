# -*- coding: utf-8 -*-

import logging
import os
import json
import numpy as np
from keras.utils import to_categorical
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Dense, TimeDistributed, Dropout
from keras.callbacks import EarlyStopping
from att import Attention
from keras.layers import GRU, LSTM, Bidirectional
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report
from operator import itemgetter

from draw_picture import draw_loss_acc
from load_data import get_train_test_pd, read_txt_file
from utils import model_dir, json_label2id, filepath, MAX_SEQ_LEN, model_name, epoch, log_path, train_file_path, \
    test_file_path, batch_size

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

# 读取文件并进行转换
# train_df, test_df = get_train_test_pd()
bert_model = BertVector(pooling_strategy="NONE", max_seq_len=MAX_SEQ_LEN)
f = lambda text: bert_model.encode([text])["encodes"][0]

# json_label2id = './data/rel_dict.json'
# 读取label2id字典
with open(json_label2id, "r", encoding="utf-8") as h:
    label_id_dict = json.loads(h.read())

id_label_dict = {v: k for k, v in label_id_dict.items()}
# print(len(label_id_dict.keys()))
num_classes = 2
train_df, test_df = get_train_test_pd()
train_df['x'] = train_df['text'].apply(f)
test_df['x'] = test_df['text'].apply(f)

# 训练集和测试集
x_test = np.array([vec for vec in test_df['x']])
y_test = np.array([vec for vec in test_df['label']])

# 将类型y值转化为ont-hot向量
y_test = to_categorical(y_test, num_classes)


class DataGenerator(object):
    def __init__(self, file_path, batch_size=batch_size, random=True):
        self.tags, self.sentences = read_txt_file(file_path)
        print("sentences length: %s " % len(self.sentences))
        print("last sentence: ", self.sentences[-1])
        print("tags length: %s " % len(self.tags))
        self.batch_size = batch_size
        self.random = random
        self.steps = len(self.sentences) // self.batch_size
        if len(self.sentences) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self, random=False):
        idxs = list(range(len(self.sentences)))
        if random:
            np.random.shuffle(idxs)
        # encoding,可能内存不足，修改为yield生成器
        x, new_y = [], []
        for i in idxs:
            sent = self.sentences[i]
            x.append(f(sent))
            tag = [self.tags[i]]
            # print(tag)
            new_y.append(tag)

            if len(x) == self.batch_size or i == idxs[-1]:
                # print('NEW_Y', new_y)
                # print('x', x)
                # 多维数组，数组的元素不为空，为随机产生的数据
                y = np.empty(shape=(len(new_y), MAX_SEQ_LEN, num_classes))
                # print(new_y[0])  # 每个句子标注的label
                # 将new_y中的元素编码成ont-hot encoding
                for j, seq in enumerate(new_y):
                    y[j, :, :] = to_categorical(seq, num_classes=num_classes)
                x = np.array(x)
                yield x, y
                x, new_y = [], []

    def forfit(self):
        while True:
            for d in self.__iter__(self.random):
                yield d
# 将类型y值转化为ont-hot向量
# num_classes = 2
# y_train = to_categorical(y_train, num_classes)
# y_test = to_categorical(y_test, num_classes)

# 模型结构：BERT + 双向GRU + Attention + FC
if model_name == 'albert':
    inputs = Input(shape=(MAX_SEQ_LEN, 312,), name="bert_output")
elif model_name == 'ELECTRA':
    inputs = Input(shape=(MAX_SEQ_LEN, 312,), name="bert_output")
elif model_name == 'roberta_wwm_large':
    inputs = Input(shape=(MAX_SEQ_LEN, 1024,), name="bert_output")
else:
    inputs = Input(shape=(MAX_SEQ_LEN, 768,), name="bert_output")
gru = Bidirectional(GRU(MAX_SEQ_LEN, return_sequences=True), name="Bi-GRU")(inputs)
drop = Dropout(0.2, name="dropout")(gru)

# TODO 此处attention在数据生成器的时候有问题，向量维度不对，暂时没有用
attention = Attention(32)(drop)

# TimeDistributed实现二维向三维的过度，在每个时间步上均操作了Dense，从而增加了模型实现一对多和多对多的能力
output = TimeDistributed(Dense(num_classes, activation='softmax'), name="time_distributed")(drop)
model = Model(inputs, output)

# 模型可视化
# from keras.utils import plot_model
# plot_model(model, to_file=r'data\loss_acc_png\model.loss_acc_png', show_shapes=True)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

# early stopping
early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, mode='max')

# 如果原来models文件夹下存在.h5文件，则全部删除
if os.listdir(model_dir):
    for file in os.listdir(model_dir):
        os.remove(os.path.join(model_dir, file))

# 保存最新的val_acc最好的模型文件
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

# 读取训练集，验证集和测试集数据
train_generator = DataGenerator(train_file_path)
# dev_generator = DataGenerator(dev_file_path)
test_generator = DataGenerator(test_file_path, random=False)
# 模型训练以及评估
history = model.fit_generator(train_generator.forfit(),
                              steps_per_epoch=len(train_generator),
                              epochs=epoch,
                              validation_data=test_generator.forfit(),
                              validation_steps=len(test_generator),
                              callbacks=[early_stopping, checkpoint],
                              verbose=1,)

# 读取关系对应表
with open(json_label2id, 'r', encoding='utf-8') as f:
    label_id_dict = json.loads(f.read())

sorted_label_id_dict = sorted(label_id_dict.items(), key=itemgetter(1))
values = [_[0] for _ in sorted_label_id_dict]
# 输出每一类的classification report
y_pred = model.predict(x_test, batch_size=batch_size)
print(classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1), target_names=values, digits=4))
draw_loss_acc(history)

logging.basicConfig(level=logging.DEBUG,  # 控制台打印的日志级别
                    filename=log_path,
                    filemode='a',  # 模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志,a是追加模式，默认如果不写的话，就是追加模式
                    format='%(asctime)s - %(levelname)s: %(message)s\n'  # 日志格式
                    )
logging.info(classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1), target_names=values, digits=4))
