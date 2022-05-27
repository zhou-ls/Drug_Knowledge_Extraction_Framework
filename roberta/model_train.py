# -*- coding: utf-8 -*-
# 模型训练
import logging
import os
import json
import numpy as np
from keras.utils import to_categorical
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Dense
from keras.callbacks import EarlyStopping
from att import Attention
from keras.layers import GRU, LSTM, Bidirectional
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report
from operator import itemgetter

from draw_picture import draw_loss_acc
from load_data import get_train_test_pd
from utils import model_dir, json_label2id, filepath, MAX_SEQ_LEN, model_name, epoch, log_path, num_classes, batch_size

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
train_df, test_df = get_train_test_pd()
bert_model = BertVector(pooling_strategy="NONE", max_seq_len=MAX_SEQ_LEN)
print('begin encoding')
f = lambda text: bert_model.encode([text])["encodes"][0]

train_df['x'] = train_df['text'].apply(f)
test_df['x'] = test_df['text'].apply(f)
print('end encoding')

# 训练集和测试集
x_train = np.array([vec for vec in train_df['x']])
x_test = np.array([vec for vec in test_df['x']])
y_train = np.array([vec for vec in train_df['label']])
y_test = np.array([vec for vec in test_df['label']])
# print('x_train: ', x_train.shape)

# 将类型y值转化为ont-hot向量
# num_classes = 2
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# 模型结构：BERT + 双向GRU + Attention + FC
if model_name == 'albert':
    inputs = Input(shape=(MAX_SEQ_LEN, 312,))  # albert_tiny,electra_tiny是[80， 312，]
elif model_name == 'ELECTRA':
    inputs = Input(shape=(MAX_SEQ_LEN, 312,))
elif model_name == 'roberta_wwm_large':
    inputs = Input(shape=(MAX_SEQ_LEN, 1024,))
else:
    inputs = Input(shape=(MAX_SEQ_LEN, 768,))
# print(inputs)  # (?, 128, 312)
gru = Bidirectional(GRU(MAX_SEQ_LEN, dropout=0.2, return_sequences=True))(inputs)
attention = Attention(32)(gru)
output = Dense(num_classes, activation='softmax')(attention)
print(output)
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

# 模型训练以及评估
history = model.fit(x_train,
                    y_train,
                    validation_data=(x_test, y_test),
                    batch_size=batch_size,
                    epochs=epoch,
                    callbacks=[early_stopping, checkpoint])

print('在测试集上的效果：', model.evaluate(x_test, y_test))

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
